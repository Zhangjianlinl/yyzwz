import os
import sys
import torch
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
import tempfile
import threading
import queue
import time
from collections import deque

sys.path.insert(0, 'Fun-ASR-Nano-2512')

class VADStreamingASR:
    def __init__(self, model_dir="Fun-ASR-Nano-2512", sample_rate=16000):
        """
        使用 VAD 的流式语音识别
        
        Args:
            model_dir: 模型目录
            sample_rate: 采样率
        """
        from funasr import AutoModel
        
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_running = False
        
        # VAD 参数
        self.vad_threshold = 500  # 能量阈值
        self.min_speech_duration = 1  # 最小语音时长（秒）
        self.max_speech_duration = 30.0  # 最大语音时长（秒）
        self.silence_duration = 2  # 静音持续时长判定为语音结束（秒）
        
        # 加载模型
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"加载模型中... (设备: {device})")
        
        model_dir = os.path.abspath(model_dir)
        self.model = AutoModel(
            model=model_dir,
            trust_remote_code=True,
            remote_code="./model.py",
            device=device,
            disable_update=True,
        )
        print("模型加载完成！\n")
    
    def is_speech(self, audio_chunk):
        """简单的 VAD：基于能量检测"""
        energy = np.abs(audio_chunk).mean()
        return energy > self.vad_threshold
    
    def audio_callback(self, indata, frames, time_info, status):
        """音频流回调函数"""
        if status:
            print(f"状态: {status}")
        self.audio_queue.put(indata.copy())
    
    def process_audio(self, language="中文", itn=True):
        """处理音频队列中的数据"""
        speech_buffer = []
        silence_frames = 0
        is_speaking = False
        frame_duration = 0.1  # 每帧 100ms
        silence_threshold = int(self.silence_duration / frame_duration)
        min_frames = int(self.min_speech_duration / frame_duration)
        max_frames = int(self.max_speech_duration / frame_duration)
        
        print("等待语音输入...")
        
        while self.is_running:
            try:
                # 获取音频数据
                audio_chunk = self.audio_queue.get(timeout=0.1)
                audio_chunk = audio_chunk.flatten()
                
                # 检测是否有语音
                has_speech = self.is_speech(audio_chunk)
                
                if has_speech:
                    if not is_speaking:
                        print("🎤 检测到语音...")
                        is_speaking = True
                    
                    speech_buffer.extend(audio_chunk)
                    silence_frames = 0
                    
                    # 防止录音过长
                    if len(speech_buffer) > max_frames * len(audio_chunk):
                        self._recognize_speech(speech_buffer, language, itn)
                        speech_buffer = []
                        is_speaking = False
                
                else:
                    if is_speaking:
                        speech_buffer.extend(audio_chunk)
                        silence_frames += 1
                        
                        # 静音持续足够长，判定语音结束
                        if silence_frames >= silence_threshold:
                            # 检查是否达到最小时长
                            if len(speech_buffer) >= min_frames * len(audio_chunk):
                                self._recognize_speech(speech_buffer, language, itn)
                            
                            speech_buffer = []
                            is_speaking = False
                            silence_frames = 0
                            print("等待语音输入...")
            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"处理错误: {e}")
                import traceback
                traceback.print_exc()
    
    def _recognize_speech(self, audio_data, language, itn):
        """识别语音"""
        try:
            print("⏳ 识别中...")
            
            # 转换为 numpy 数组
            audio_array = np.array(audio_data, dtype=np.int16)
            
            # 保存为临时文件
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
            
            finally:
                # 删除临时文件
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        
        except Exception as e:
            print(f"识别错误: {e}")
    
    def start_streaming(self, language="中文", itn=True):
        """开始流式识别"""
        self.is_running = True
        
        # 启动音频处理线程
        process_thread = threading.Thread(
            target=self.process_audio,
            args=(language, itn),
            daemon=True
        )
        process_thread.start()
        
        # 启动结果显示线程
        result_thread = threading.Thread(
            target=self._display_results,
            daemon=True
        )
        result_thread.start()
        
        # 启动音频流
        print("=" * 60)
        print("流式语音识别已启动")
        print(f"语言: {language} | 文本规整: {'是' if itn else '否'}")
        print("=" * 60)
        print("\n按 Ctrl+C 停止\n")
        
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='int16',
            callback=self.audio_callback,
            blocksize=int(self.sample_rate * 0.1)  # 100ms 块
        ):
            try:
                while self.is_running:
                    time.sleep(0.1)
            
            except KeyboardInterrupt:
                print("\n\n停止识别...")
                self.is_running = False
    
    def _display_results(self):
        """显示识别结果"""
        while self.is_running:
            try:
                result = self.result_queue.get(timeout=0.1)
                print(f"\n✅ [{time.strftime('%H:%M:%S')}] {result}\n")
            except queue.Empty:
                continue
    
    def stop(self):
        """停止流式识别"""
        self.is_running = False


def main():
    print("初始化流式语音识别系统...")
    print("提示: 说话时会自动检测并识别，静音 0.8 秒后自动结束识别\n")
    
    # 选择语言
    print("请选择识别语言:")
    print("1. 中文（普通话）")
    print("2. 英文")
    print("3. 日文")
    print("4. 粤语（需要 Fun-ASR-MLT-Nano-2512 模型）")
    
    choice = input("\n请输入选项 (1-4，默认1): ").strip() or "1"
    
    language_map = {
        "1": "中文",
        "2": "English",
        "3": "日文",
        "4": "粤语"
    }
    
    language = language_map.get(choice, "中文")
    
    # 如果选择粤语，提示需要多语言模型
    if choice == "4":
        print("\n⚠️  注意: 粤语识别需要 Fun-ASR-MLT-Nano-2512 模型")
        print("当前使用的是 Fun-ASR-Nano-2512，可能无法正确识别粤语")
        print("如需使用多语言模型，请下载并修改 model_dir 参数\n")
        cont = input("是否继续? (y/n): ").strip().lower()
        if cont != 'y':
            return
    
    # 创建流式识别器
    asr = VADStreamingASR(
        model_dir="Fun-ASR-Nano-2512",
        sample_rate=16000
    )
    
    # 可以调整 VAD 参数
    # asr.vad_threshold = 300  # 降低阈值，更敏感
    # asr.silence_duration = 1.0  # 增加静音时长
    
    # 开始流式识别
    asr.start_streaming(language=language, itn=True)


if __name__ == "__main__":
    main()
