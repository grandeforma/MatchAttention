import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import os
import threading
from PIL import Image
from glob import glob

from dataloader.stereo import transforms
from utils.utils import InputPadder, calc_noc_mask
from utils.file_io import write_pfm
from models.match_stereo import MatchStereo
import gc


torch.backends.cudnn.benchmark = True

class Args:
    def __init__(self, variant='small', mat_impl='pytorch', scale=1.0, precision='fp32', device_id=0):
        self.variant = variant
        self.mat_impl = mat_impl
        self.scale = scale
        self.precision = precision
        self.device_id = device_id


def run_frame(model, left, right, stereo, low_res_init, factor=2.):
    if low_res_init: # downsample to 1/2, can also be 1/4
        left_ds = F.interpolate(left, scale_factor=1/factor, mode='bilinear', align_corners=True)
        right_ds = F.interpolate(right, scale_factor=1/factor, mode='bilinear', align_corners=True)
        padder_ds = InputPadder(left_ds.shape, padding_factor=32)
        left_ds, right_ds = padder_ds.pad(left_ds, right_ds)

        field_up_ds = model(left_ds, right_ds, stereo=stereo)['field_up']
        field_up_ds = padder_ds.unpad(field_up_ds.permute(0, 3, 1, 2).contiguous()).contiguous()
        field_up_init = F.interpolate(field_up_ds, scale_factor=factor/32, mode='bilinear', align_corners=True)*(factor/32) # init resolution 1/32
        field_up_init = field_up_init.permute(0, 2, 3, 1).contiguous()
        results_dict = model(left, right, stereo=stereo, init_flow=field_up_init)
    else:
        results_dict = model(left, right, stereo=stereo)

    return results_dict

class Stereo2Depth:
    def __init__(self, checkpoint_path, device_id=0, precision='fp32', variant='small', scale=1.0):
        self.scale = scale
        self.device = torch.device(f'cuda:{device_id}') if torch.cuda.is_available() and device_id >=0 else 'cpu'
        dtypes = {'fp32': torch.float, 'fp16': torch.half, 'bf16': torch.bfloat16}
        self.dtype = dtypes[precision]

        self.args = Args(variant=variant, mat_impl='pytorch', scale=scale, precision=precision, device_id=device_id)
        self.model = MatchStereo(self.args)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(state_dict=checkpoint['model'], strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.model = self.model.to(self.dtype)
        self.val_transform = transforms.Compose([
            transforms.Resize(scale_x=scale, scale_y=scale),
            transforms.ToTensor(no_normalize=True),
        ])

    def run(self, left_img, right_img):
        scale = self.scale
        dtype = self.dtype
        model = self.model

        gc.collect()
        torch.cuda.empty_cache()


        """Run MatchStereo/MatchFlow on stereo/flow pairs"""

        # if torch.cuda.is_available() and not args.no_compile and args.device_id >=0:
        #     print('compiling the model, this may take several minutes')
        #     torch.backends.cuda.matmul.allow_tf32 = True
        #     model = torch.compile(model, dynamic=False)

        # if args.middv3_dir is not None:
        #     left_names = sorted(glob(args.middv3_dir + '/*/*/im0.png'))
        #     right_names = sorted(glob(args.middv3_dir + '/*/*/im1.png'))
        # elif args.eth3d_dir is not None:
        #     left_names = sorted(glob(args.eth3d_dir + '/*/*/im0.png'))
        #     right_names = sorted(glob(args.eth3d_dir + '/*/*/im1.png'))
        # else:
        #     left_names = sorted(glob(args.img0_dir + '/*.png') + glob(args.img0_dir + '/*.jpg') + glob(args.img0_dir + '/*.bmp'))
        #     right_names = sorted(glob(args.img1_dir + '/*.png') + glob(args.img1_dir + '/*.jpg') + glob(args.img1_dir + '/*.bmp'))
        # assert len(left_names) == len(right_names)

        #num_samples = len(left_names)
        #print('%d test samples found' % num_samples)

        # if torch.cuda.is_available() and args.device_id >=0:
        #     start_event = torch.cuda.Event(enable_timing=True)
        #     end_event = torch.cuda.Event(enable_timing=True)
        # else:
        #     args.test_inference_time = False

        left = left_img #np.array(Image.open(left_names[i]).convert('RGB')).astype(np.float32)
        right = right_img #np.array(Image.open(right_names[i]).convert('RGB')).astype(np.float32)

        sample = {'left': left, 'right': right}
        sample = self.val_transform(sample)
        left = sample['left'].to(self.device, dtype=self.dtype).unsqueeze(0) # [1, 3, H, W]
        right = sample['right'].to(self.device, dtype=self.dtype).unsqueeze(0) # [1, 3, H, W]

        #if args.inference_size is None:
        padder = InputPadder(left.shape, padding_factor=32)
        left, right = padder.pad(left, right)
        #else:
        #    ori_size = left.shape[-2:]

        #    left = F.interpolate(left, size=args.inference_size, mode='bilinear', align_corners=True)
        #    right = F.interpolate(right, size=args.inference_size, mode='bilinear', align_corners=True)

        print("Resolution: ", left.shape)
        with torch.inference_mode():
            # if args.test_inference_time:
            #     for _ in range(5): # warmup
            #         _ = model(left, right, stereo=stereo)

            #     start_event.record()
            #     for _ in range(5):
            #         results_dict = run_frame(model, left, right, stereo, args.low_res_init)
            #     end_event.record()
            #     end_event.synchronize()

            #     inference_time = start_event.elapsed_time(end_event) / 5  # in milliseconds
            #     print(f"Inference Time (GPU) on {left_names[i]}: {inference_time:.6f} ms")

            #else:
            results_dict = run_frame(model, left, right, True, False)

            field_up = results_dict['field_up'].permute(0, 3, 1, 2).float().contiguous()
            self_rpos = results_dict['self_rpos'].permute(0, 3, 1, 2).float().contiguous()
            self_rpos = F.interpolate(self_rpos, scale_factor=4, mode='bilinear', align_corners=True)*4
            # if args.inference_size is None:
            field_up = padder.unpad(field_up)
            self_rpos = padder.unpad(self_rpos)
            # else:
            #     field_up = F.interpolate(field_up, size=ori_size, mode='bilinear', align_corners=True)
            #     field_up[:, 0] = field_up[:, 0] * (ori_size[1] / float(args.inference_size[1]))
            #     field_up[:, 1] = field_up[:, 1] * (ori_size[0] / float(args.inference_size[0]))

            #     self_rpos = F.interpolate(self_rpos, size=ori_size, mode='bilinear', align_corners=True)
            #     self_rpos[:, 0] = self_rpos[:, 0] * ori_size[1] / float(args.inference_size[1])
            #     self_rpos[:, 1] = self_rpos[:, 1] * ori_size[0] / float(args.inference_size[0])

        # if args.middv3_dir is not None:
        #     save_name = left_names[i].replace('/MiddEval3', '/MiddEval3_results').replace('/im0.png', '/disp0MatchStereo.pfm')
        # elif args.eth3d_dir is not None:
        #     parts = list(Path(left_names[i]).parts)
        #     parts[1] = "ETH3D_results"
        #     parts[2] = "low_res_two_view"
        #     save_name = str(Path(*parts[:3]) / f"{parts[3]}.pfm")
        # else:
        #     save_name = os.path.join(args.output_path, os.path.basename(left_names[i])[:-4] + f'_{args.mode}.pfm')
        # os.makedirs(os.path.dirname(save_name), exist_ok=True)

        #noc_mask = calc_noc_mask(field_up.permute(0, 2, 3, 1), A=8)
        ## field[~noc_mask] = torch.inf # NOTE: can filter out un-reliable matches by consistency check
        #noc_mask = noc_mask[0].detach().cpu().numpy()
        #noc_mask = np.where(noc_mask, 255, 128).astype(np.uint8)
        #noc_img = Image.fromarray(noc_mask)
        #noc_img.save(save_name[:-4] + '_noc.png')
        field_up = torch.cat((field_up, torch.zeros_like(field_up[:, :1])), dim=1)
        field_up = field_up.permute(0, 2, 3, 1).contiguous() # [B, H, W, 3]
        field, field_r = field_up.chunk(2, dim=0)
        #if stereo:
        field = (-field[..., 0]).clamp(min=0)
        field_r = field_r[..., 0].clamp(min=0)
        field = field[0].detach().cpu().numpy()
        return field
        #field_r = field_r[0].detach().cpu().numpy()
        # write_pfm(save_name, field)
        # if args.save_right:
        #     write_pfm(save_name[:-4] + '_r.pfm', field_r)

        # if args.save_rpos:
        #     self_rpos, _ = self_rpos.chunk(2, dim=0)
        #     self_rpos = self_rpos[0].detach().cpu().numpy()
        #     write_pfm(save_name[:-4] + '_self_rpos_x.pfm', self_rpos[0])
        #     write_pfm(save_name[:-4] + '_self_rpos_y.pfm', self_rpos[1])

        # if args.test_inference_time:
        #     if args.middv3_dir is not None:
        #         save_time_name = save_name.replace('/disp0MatchStereo.pfm', '/timeMatchStereo.txt')
        #         with open(save_time_name, 'w') as f:
        #             f.write(str(inference_time / 1000))
        #     elif args.eth3d_dir is not None:
        #         save_time_name = save_name.replace('.pfm', '.txt')
        #         with open(save_time_name, 'w') as f:
        #             f.write(str(f"runtime {inference_time / 1000}"))

        print("Inference done.")


class Stereo2DepthWorker:
    """Run Stereo2Depth inference on a background thread.

    The worker stores only the newest pending input and newest finished
    disparity so the caller can continue capturing frames without blocking on
    inference.
    """

    def __init__(self, stereo2depth):
        self.stereo2depth = stereo2depth
        self._job_condition = threading.Condition()
        self._latest_job = None
        self._closed = False
        self._result_lock = threading.Lock()
        self._latest_result = None
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def submit(self, left_img, right_img):
        with self._job_condition:
            if self._closed:
                raise RuntimeError("Stereo2DepthWorker has been closed")
            self._latest_job = (
                np.array(left_img, copy=True),
                np.array(right_img, copy=True),
            )
            self._job_condition.notify()

    def get_latest_result(self):
        with self._result_lock:
            latest = self._latest_result
            self._latest_result = None
            return latest

    def close(self):
        with self._job_condition:
            self._closed = True
            self._job_condition.notify()
        self._thread.join()

    def _worker_loop(self):
        while True:
            with self._job_condition:
                while self._latest_job is None and not self._closed:
                    self._job_condition.wait()

                if self._closed:
                    break

                left_img, right_img = self._latest_job
                self._latest_job = None

            disparity = self.stereo2depth.run(left_img, right_img)
            with self._result_lock:
                self._latest_result = disparity

def run(args):
    """Run MatchStereo/MatchFlow on stereo/flow pairs"""
    stereo = (args.mode == 'stereo')
    val_transform_list = [transforms.Resize(scale_x=args.scale, scale_y=args.scale), 
                          transforms.ToTensor(no_normalize=True)]
    val_transform = transforms.Compose(val_transform_list)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    device = torch.device(f'cuda:{args.device_id}') if torch.cuda.is_available() and args.device_id >=0 else 'cpu'
    dtypes = {'fp32': torch.float, 'fp16': torch.half, 'bf16': torch.bfloat16}
    dtype = dtypes[args.precision]

    model = MatchStereo(args)
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict=checkpoint['model'], strict=False)
    model.to(device)
    model.eval()
    model = model.to(dtype)
    if torch.cuda.is_available() and not args.no_compile and args.device_id >=0:
        print('compiling the model, this may take several minutes')
        torch.backends.cuda.matmul.allow_tf32 = True
        model = torch.compile(model, dynamic=False)

    if args.middv3_dir is not None:
        left_names = sorted(glob(args.middv3_dir + '/*/*/im0.png'))
        right_names = sorted(glob(args.middv3_dir + '/*/*/im1.png'))
    elif args.eth3d_dir is not None:
        left_names = sorted(glob(args.eth3d_dir + '/*/*/im0.png'))
        right_names = sorted(glob(args.eth3d_dir + '/*/*/im1.png'))
    else:
        left_names = sorted(glob(args.img0_dir + '/*.png') + glob(args.img0_dir + '/*.jpg') + glob(args.img0_dir + '/*.bmp'))
        right_names = sorted(glob(args.img1_dir + '/*.png') + glob(args.img1_dir + '/*.jpg') + glob(args.img1_dir + '/*.bmp'))
    assert len(left_names) == len(right_names)

    num_samples = len(left_names)
    print('%d test samples found' % num_samples)

    if torch.cuda.is_available() and args.device_id >=0:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
    else:
        args.test_inference_time = False

    for i in range(num_samples):

        left = np.array(Image.open(left_names[i]).convert('RGB')).astype(np.float32)
        right = np.array(Image.open(right_names[i]).convert('RGB')).astype(np.float32)

        sample = {'left': left, 'right': right}
        sample = val_transform(sample)
        left = sample['left'].to(device, dtype=dtype).unsqueeze(0) # [1, 3, H, W]
        right = sample['right'].to(device, dtype=dtype).unsqueeze(0) # [1, 3, H, W]

        if args.inference_size is None:
            padder = InputPadder(left.shape, padding_factor=32)
            left, right = padder.pad(left, right)
        else:
            ori_size = left.shape[-2:]

            left = F.interpolate(left, size=args.inference_size, mode='bilinear', align_corners=True)
            right = F.interpolate(right, size=args.inference_size, mode='bilinear', align_corners=True)

        print("Resolution: ", left.shape)
        with torch.inference_mode():
            if args.test_inference_time:
                for _ in range(5): # warmup
                    _ = model(left, right, stereo=stereo)

                start_event.record()
                for _ in range(5):
                    results_dict = run_frame(model, left, right, stereo, args.low_res_init)
                end_event.record()
                end_event.synchronize()

                inference_time = start_event.elapsed_time(end_event) / 5  # in milliseconds
                print(f"Inference Time (GPU) on {left_names[i]}: {inference_time:.6f} ms")

            else:
                results_dict = run_frame(model, left, right, stereo, args.low_res_init)

            field_up = results_dict['field_up'].permute(0, 3, 1, 2).float().contiguous()
            self_rpos = results_dict['self_rpos'].permute(0, 3, 1, 2).float().contiguous()
            self_rpos = F.interpolate(self_rpos, scale_factor=4, mode='bilinear', align_corners=True)*4
            if args.inference_size is None:
                field_up = padder.unpad(field_up)
                self_rpos = padder.unpad(self_rpos)
            else:
                field_up = F.interpolate(field_up, size=ori_size, mode='bilinear', align_corners=True)
                field_up[:, 0] = field_up[:, 0] * (ori_size[1] / float(args.inference_size[1]))
                field_up[:, 1] = field_up[:, 1] * (ori_size[0] / float(args.inference_size[0]))

                self_rpos = F.interpolate(self_rpos, size=ori_size, mode='bilinear', align_corners=True)
                self_rpos[:, 0] = self_rpos[:, 0] * ori_size[1] / float(args.inference_size[1])
                self_rpos[:, 1] = self_rpos[:, 1] * ori_size[0] / float(args.inference_size[0])

        if args.middv3_dir is not None:
            save_name = left_names[i].replace('/MiddEval3', '/MiddEval3_results').replace('/im0.png', '/disp0MatchStereo.pfm')
        elif args.eth3d_dir is not None:
            parts = list(Path(left_names[i]).parts)
            parts[1] = "ETH3D_results"
            parts[2] = "low_res_two_view"
            save_name = str(Path(*parts[:3]) / f"{parts[3]}.pfm")
        else:
            save_name = os.path.join(args.output_path, os.path.basename(left_names[i])[:-4] + f'_{args.mode}.pfm')
        os.makedirs(os.path.dirname(save_name), exist_ok=True)

        noc_mask = calc_noc_mask(field_up.permute(0, 2, 3, 1), A=8)
        ## field[~noc_mask] = torch.inf # NOTE: can filter out un-reliable matches by consistency check
        noc_mask = noc_mask[0].detach().cpu().numpy()
        noc_mask = np.where(noc_mask, 255, 128).astype(np.uint8)
        noc_img = Image.fromarray(noc_mask)
        noc_img.save(save_name[:-4] + '_noc.png')
        field_up = torch.cat((field_up, torch.zeros_like(field_up[:, :1])), dim=1)
        field_up = field_up.permute(0, 2, 3, 1).contiguous() # [B, H, W, 3]
        field, field_r = field_up.chunk(2, dim=0)
        if stereo:
            field = (-field[..., 0]).clamp(min=0)
            field_r = field_r[..., 0].clamp(min=0)
        field = field[0].detach().cpu().numpy()
        field_r = field_r[0].detach().cpu().numpy()
        print(field)
        write_pfm(save_name, field)
        if args.save_right:
            write_pfm(save_name[:-4] + '_r.pfm', field_r)

        if args.save_rpos:
            self_rpos, _ = self_rpos.chunk(2, dim=0)
            self_rpos = self_rpos[0].detach().cpu().numpy()
            write_pfm(save_name[:-4] + '_self_rpos_x.pfm', self_rpos[0])
            write_pfm(save_name[:-4] + '_self_rpos_y.pfm', self_rpos[1])

        if args.test_inference_time:
            if args.middv3_dir is not None:
                save_time_name = save_name.replace('/disp0MatchStereo.pfm', '/timeMatchStereo.txt')
                with open(save_time_name, 'w') as f:
                    f.write(str(inference_time / 1000))
            elif args.eth3d_dir is not None:
                save_time_name = save_name.replace('.pfm', '.txt')
                with open(save_time_name, 'w') as f:
                    f.write(str(f"runtime {inference_time / 1000}"))

    print("Inference done.")

def main():
    """Run MatchStereo/MatchFlow inference example"""
    parser = argparse.ArgumentParser(
        description="Inference scripts of MatchStereo/MatchFlow with PyTorch models."
    )
    parser.add_argument('--checkpoint_path', required=True, type=str, help='Path to MatchStereo/MatchFlow checkpoint')
    parser.add_argument('--mode', choices=['stereo', 'flow'], default='stereo', help='Support stereo and flow tasks')
    parser.add_argument('--img0_dir', default=None, type=str, help='Reference view')
    parser.add_argument('--img1_dir', default=None, type=str, help='Target view')
    parser.add_argument('--middv3_dir', default=None, type=str)
    parser.add_argument('--eth3d_dir', default=None, type=str)
    parser.add_argument('--output_path', default='outputs', type=str)
    parser.add_argument('--device_id', default=0, type=int, help='Devide id of gpu, -1 for cpu')
    parser.add_argument('--scale', default=1, type=float)
    parser.add_argument('--inference_size', default=None, type=int, nargs='+', help='Shall be divisible by 32')
    parser.add_argument('--mat_impl', choices=['pytorch', 'cuda'], default='pytorch', help='MatchAttention implementation')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument('--variant', choices=['tiny', 'small', 'base'], default='tiny')
    parser.add_argument('--no_compile', action='store_true', default=False, help='Disable torch.compile')
    parser.add_argument('--test_inference_time', action='store_true', default=False)
    parser.add_argument('--save_right', action='store_true', default=False, help='Save the right/target view disp/flow')
    parser.add_argument('--save_rpos', action='store_true', default=False, help='Save the self relative positions')
    parser.add_argument('--low_res_init', action='store_true', default=False, help='Low-resolution init, use this when image is of high-resolution (>=2K)')

    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
