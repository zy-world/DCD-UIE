from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import os
import json
import torch
import clip


class LRHRDataset(Dataset):
    def __init__(self, dataroot_HR, dataroot_LR, datatype, l_resolution=16, r_resolution=128, split='train',
                 data_len=-1, need_LR=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split
        self.text_features = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)

        base_dir_for_text = os.path.dirname(dataroot_HR)
        self.text_dir = os.path.join(base_dir_for_text, f"text-EUVP-200-{split}")

        if datatype == 'lmdb':
            self.env = lmdb.open(dataroot_HR, readonly=True, lock=False,
                                 readahead=False, meminit=False)
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get("length".encode("utf-8")))
        elif datatype == 'img':
            self.hr_path = Util.get_paths_from_images(dataroot_HR)
            self.sr_path = Util.get_paths_from_images(dataroot_LR)

            self._match_pairs()

            if self.need_LR:
                self.lr_path = Util.get_paths_from_images(dataroot_LR)
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized.'.format(datatype))

        self.dataset_len = len(self.hr_path)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

        if self.data_len == 0:
            raise ValueError(
                f"CRITICAL ERROR: No data found. Please check paths:\nHR: {dataroot_HR}\nSR/LR: {dataroot_LR}")

    def _match_pairs(self):
        sr_map = {os.path.splitext(os.path.basename(p))[0]: p for p in self.sr_path}
        hr_map = {os.path.splitext(os.path.basename(p))[0]: p for p in self.hr_path}

        matched_sr = []
        matched_hr = []
        for hr_name, hr_path in hr_map.items():
            if hr_name in sr_map:
                matched_hr.append(hr_path)
                matched_sr.append(sr_map[hr_name])

        self.hr_path = sorted(matched_hr)
        self.sr_path = sorted(matched_sr)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_SR = None

        if self.datatype == 'lmdb':
            pass
        else:
            index = index % len(self.hr_path)
            img_HR = Image.open(self.hr_path[index]).convert("RGB")
            img_SR = Image.open(self.sr_path[index]).convert("RGB")
        img_name = os.path.splitext(os.path.basename(self.hr_path[index]))[0]
        json_path = os.path.join(self.text_dir, f"{img_name}.json")
        text_desc = "a beautiful underwater photo"
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                try:
                    text_data = json.load(f)
                    text_desc = text_data["caption"]
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON for {json_path}")
        with torch.no_grad():
            text_tokens = clip.tokenize([text_desc]).to(self.device)
            text_feature = self.clip_model.encode_text(text_tokens).squeeze(0).cpu()

        [img_SR, img_HR] = Util.transform_augment(
            [img_SR, img_HR], split=self.split, min_max=(-1, 1))
        return {'HR': img_HR, 'SR': img_SR, 'text_feature': text_feature, 'Index': index}