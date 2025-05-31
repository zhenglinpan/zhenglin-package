import os
import torch
from typing import List, Union

from .cv import cv

class vd:
    def __init__(self, frames: List[cv]):
        self.frames = frames

    def __getitem__(self, idx):
        if isinstance(idx, (int, slice)):
            return self.frames[idx]
        elif isinstance(idx, tuple):
            return [self.frames[i] for i in idx]
        elif isinstance(idx, list):
            return [self.frames[i] for i in idx]
        else:
            raise TypeError("Unsupported index type.")

    def __len__(self):
        return len(self.frames)

    @property
    def tensor(self):
        tensors = [f.tensor for f in self.frames]
        return torch.stack(tensors, dim=1)  # shape: [1, T, C, H, W]

    @staticmethod
    def read_video(path_in: str, frames: Union[int, List[int]] = None, img_type: str = 'cv2'):
        from decord import VideoReader, cpu
        
        vr = VideoReader(path_in, ctx=cpu(0))
        total_frames = len(vr)

        if frames is None:
            indices = list(range(total_frames))
        elif isinstance(frames, int):
            indices = [frames]
        else:
            indices = frames

        frames_out = [cv(vr[i].asnumpy()) for i in indices]

        if img_type == 'tensor':
            tensors = [f.tensor for f in frames_out]
            return torch.stack(tensors, dim=1) / 255.  # shape: [1, T, C, H, W]

        if isinstance(frames, int):
            return frames_out[0]
        
        return vd(frames_out)

    @staticmethod
    def save_video(path_out: str, obj, fps=30, serialize=False, lossless=False, **kwargs):
        import cv2
        
        if isinstance(obj, str):
            path_out, obj = obj, path_out

        frames = []

        if isinstance(obj, vd):
            frames = [f.cv2 for f in obj.frames]
        elif isinstance(obj, list) and all(isinstance(f, cv) for f in obj):
            frames = [f.cv2 for f in obj]
        elif isinstance(obj, torch.Tensor):  # shape: [1, T, C, H, W]
            if obj.dim() == 5:
                obj = obj.squeeze(0)
            frames = [cv(f).cv2 for f in obj]
        else:
            raise TypeError("Unsupported input for vwrite.")

        if serialize:
            os.makedirs(path_out, exist_ok=True)
            for i, frame in enumerate(frames):
                out_path = os.path.join(path_out, f'{i:05d}.png')
                cv.imwrite(frame, out_path)
            return

        if lossless:
            import subprocess
            import tempfile

            path_out = f'{os.path.splitext(path_out)[0]}.mp4'

            with tempfile.TemporaryDirectory() as temp_folder:
                for i, frame in enumerate(frames):
                    cv.imwrite(frame, os.path.join(temp_folder, f"frame_{i:04d}.png"))

                cmd = (
                    f"ffmpeg -y -framerate {fps} "
                    f"-i {temp_folder}/frame_%04d.png "
                    f"-c:v libx264 -preset veryslow -crf 0 "
                    f"{path_out}"
                )
                subprocess.run(cmd, shell=True)
        else:
            h, w = frames[0].shape[:2]
            if path_out.endswith('mp4'):
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            elif path_out.endswith('webm'):
                fourcc = cv2.VideoWriter_fourcc(*'vp80')
            else:
                raise ValueError(f"Unsupported format: {format}")

            writer = cv2.VideoWriter(path_out, fourcc, fps, (w, h))
            for frame in frames:
                writer.write(frame[:, :, ::-1])  # Convert RGB to BGR
            writer.release()


if __name__=="__main__":

    vid = vd.read_video(r"D:\stable-diffusion-webui-master\training-picker\videos\4166-Scene-002.mp4")      # full video
    
    clip1 = vid[0, 2, 4]
    clip2 = vid[:5]
    clip3 = vid[[1, 3, 5, 7]]

    vid_tensor = vid.tensor
    print(vid_tensor.shape) # torch.Size([1, 21, 3, 720, 1280])

    vd.save_video('out.mp4', vid_tensor, serialize=False, fps=30)
    