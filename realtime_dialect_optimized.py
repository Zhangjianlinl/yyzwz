import os
import sys
import torch
import subprocess
import numpy as np
from scipy.io import wavfile
import tempfile
import threading
import queue
import time

sys.path.insert(0, 'Fun-ASR-Nano-2512')

class OptimizedDialectASR:
    def __init__(self, model_dir="Fun-ASR-Nano-2512", sample_rate=16000):
        """
        优化的方言识别系统
        """
        from funasr import AutoModel

        self.sample_rate = sample_rate
        self.capture_rate = sample_rate  # 实际采集率，选设备后可能更新
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_running = False
        
        # 优化的 VAD 参数 - 更宽松，避免截断
        self.vad_threshold = 200  # 降低阈值，更敏感
        self.min_speech_duration = 0.3  # 减少最小时长
        self.max_speech_duration = 30.0  # 增加最大时长，允许更长的句子
        self.silence_duration = 2  # 增加静音时长，避免过早截断
        
        # 加载模型
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"加载方言识别模型... (设备: {device})")
        
        model_dir = os.path.abspath(model_dir)
        self.model = AutoModel(
            model=model_dir,
            trust_remote_code=True,
            remote_code="./model.py",
            device=device,
            disable_update=True,
        )
        print("✅ 模型加载完成！\n")
    
    def is_speech(self, audio_chunk):
        """改进的 VAD：基于能量和过零率"""
        # 能量检测
        energy = np.abs(audio_chunk).mean()
        
        # 过零率检测（帮助区分语音和噪音）
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_chunk)))) / (2 * len(audio_chunk))
        
        # 组合判断
        has_energy = energy > self.vad_threshold
        has_speech_characteristics = zero_crossings > 0.01  # 语音通常有较高的过零率
        
        return has_energy and has_speech_characteristics
    
    def audio_callback(self, indata, frames, time_info, status):
        """音频流回调函数（保留兼容性，arecord模式不使用）"""
        pass

    def _arecord_capture(self, alsa_device):
        """用 parec 持续采集音频块并放入队列"""
        chunk_samples = int(self.sample_rate * 0.1)
        cmd = [
            'parec',
            '--device', alsa_device,
            '--rate', str(self.sample_rate),
            '--channels', '1',
            '--format', 's16le',
            '--latency-msec', '100',
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self._arecord_proc = proc
        bytes_per_chunk = chunk_samples * 2
        buf = b''
        print(f"[debug] parec 启动，pid={proc.pid}")
        try:
            while self.is_running:
                data = proc.stdout.read(1)
                if not data:
                    err = proc.stderr.read()
                    print(f"[debug] parec 退出，stderr: {err}")
                    break
                buf += data
                if len(buf) >= bytes_per_chunk:
                    chunk = np.frombuffer(buf[:bytes_per_chunk], dtype=np.int16)
                    energy = np.abs(chunk).mean()
                    if len(buf) == bytes_per_chunk:  # 第一次
                        print(f"[debug] 收到数据，energy={energy:.0f}")
                    self.audio_queue.put(chunk.copy())
                    buf = buf[bytes_per_chunk:]
        finally:
            proc.terminate()
    
    def process_audio(self, language="中文", itn=True):
        """处理音频队列"""
        speech_buffer = []
        silence_frames = 0
        is_speaking = False
        frame_duration = 0.1
        silence_threshold = int(self.silence_duration / frame_duration)
        min_frames = int(self.min_speech_duration / frame_duration)
        max_frames = int(self.max_speech_duration / frame_duration)
        
        print("🎤 等待语音输入...\n")
        
        while self.is_running:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
                audio_chunk = audio_chunk.flatten()
                
                has_speech = self.is_speech(audio_chunk)
                
                if has_speech:
                    if not is_speaking:
                        print("🔴 录音中...", end="", flush=True)
                        is_speaking = True
                    else:
                        print(".", end="", flush=True)
                    
                    speech_buffer.extend(audio_chunk)
                    silence_frames = 0
                    
                    # 防止录音过长
                    if len(speech_buffer) > max_frames * len(audio_chunk):
                        print(f"\n⏱️  达到最大时长 ({self.max_speech_duration}秒)，开始识别...")
                        self._recognize_speech(speech_buffer, language, itn)
                        speech_buffer = []
                        is_speaking = False
                
                else:
                    if is_speaking:
                        speech_buffer.extend(audio_chunk)
                        silence_frames += 1
                        
                        # 静音持续足够长
                        if silence_frames >= silence_threshold:
                            if len(speech_buffer) >= min_frames * len(audio_chunk):
                                print(f"\n⏳ 识别中...")
                                self._recognize_speech(speech_buffer, language, itn)
                            else:
                                print(f"\n⚠️  录音太短，已忽略")
                            
                            speech_buffer = []
                            is_speaking = False
                            silence_frames = 0
                            print("\n🎤 等待语音输入...\n")
            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"\n❌ 处理错误: {e}")
                import traceback
                traceback.print_exc()
    
    def _recognize_speech(self, audio_data, language, itn):
        """识别语音"""
        try:
            audio_array = np.array(audio_data, dtype=np.int16)
            
            # 检查音频质量
            energy = np.abs(audio_array).mean()
            duration = len(audio_array) / self.sample_rate
            
            print(f"   音频时长: {duration:.2f}秒 | 平均能量: {energy:.0f}")
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                wavfile.write(tmp_path, self.sample_rate, audio_array)
            
            try:
                # 识别
                res = self.model.generate(
                    input=[tmp_path],
                    cache={},
                    batch_size=1,
                    language=language,
                    itn=itn,
                )
                
                text = res[0]["text"].strip()
                if text:
                    self.result_queue.put(text)
                else:
                    print("   ⚠️  未识别到文字")
            
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        
        except Exception as e:
            print(f"   ❌ 识别错误: {e}")
    
    def start_streaming(self, language="中文", itn=True):
        """开始流式识别"""
        self.is_running = True
        
        process_thread = threading.Thread(
            target=self.process_audio,
            args=(language, itn),
            daemon=True
        )
        process_thread.start()
        
        result_thread = threading.Thread(
            target=self._display_results,
            daemon=True
        )
        result_thread.start()
        
        print("=" * 70)
        print("🗣️  优化版方言识别系统")
        print(f"语言: {language} | 文本规整: {'是' if itn else '否'}")
        print(f"VAD 灵敏度: {self.vad_threshold} | 静音判定: {self.silence_duration}秒")
        print("=" * 70)
        print("\n💡 使用提示:")
        print("   • 请在安静环境下使用，靠近麦克风说话")
        print("   • 说完一句话后停顿 1.2 秒，系统会自动识别")
        print("   • 如果识别不准，可以说慢一点、清晰一点")
        print("   • 按 Ctrl+C 停止\n")
        
        pulse_device = 'alsa_input.usb-Sonix_Technology_Co.__Ltd._USB_2.0_Camera-02.mono-fallback'

        # 启动采集线程
        print("[debug] 启动采集线程...")
        capture_thread = threading.Thread(
            target=self._arecord_capture,
            args=(pulse_device,),
            daemon=True
        )
        capture_thread.start()
        print("[debug] 采集线程已启动")

        try:
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\n👋 停止识别...")
            self.is_running = False
    
    def _display_results(self):
        """显示识别结果"""
        while self.is_running:
            try:
                result = self.result_queue.get(timeout=0.1)
                print(f"\n✅ 识别结果: {result}\n")
            except queue.Empty:
                continue


def main():
    print("\n" + "=" * 70)
    print("🗣️  中国方言优化识别系统")
    print("=" * 70)
    print("\n📌 重要说明:")
    print("   • 模型会将方言转写成普通话文字")
    print("   • 例如：说粤语 → 输出普通话文字")
    print("   • 方言识别准确率约 70-75%（比普通话低）")
    print("\n支持的方言:")
    print("   • 7 大方言: 吴语、粤语、闽语、客家话、赣语、湘语、晋语")
    print("   • 26 种口音: 四川、重庆、河南、陕西、湖北等")
    print("=" * 70)
    
    print("\n请选择识别语言:")
    print("1. 中文（自动识别方言，转写成普通话）")
    print("2. 英文")
    print("3. 日文")
    
    choice = input("\n请输入 (1-3，默认 1): ").strip() or "1"
    
    language_map = {"1": "中文", "2": "English", "3": "日文"}
    language = language_map.get(choice, "中文")
    
    itn_choice = input("是否启用文本规整? (y/n, 默认 y): ").strip().lower()
    itn = itn_choice != 'n'
    
    # 创建识别器
    asr = OptimizedDialectASR(
        model_dir="Fun-ASR-Nano-2512",
        sample_rate=16000
    )
    
    # 高级设置（可选）
    print("\n是否调整灵敏度? (y/n, 默认 n): ", end="")
    if input().strip().lower() == 'y':
        print(f"当前 VAD 阈值: {asr.vad_threshold}")
        new_threshold = input("输入新阈值 (100-1000，默认 200): ").strip()
        if new_threshold.isdigit():
            asr.vad_threshold = int(new_threshold)
            print(f"✅ 已设置为: {asr.vad_threshold}")
    
    # 开始识别
    asr.start_streaming(language=language, itn=itn)


if __name__ == "__main__":
    main()
