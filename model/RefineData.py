from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import random

import torch
from torch.utils.data import Dataset
from PIL import Image


VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class RefineData(Dataset):
    def __init__(
        self,
        image_dir: str,
        num_samples_per_image: int = 1,
        image_size: Tuple[int, int] = (256, 256),
        object_detector=None,
        transform=None,
        return_stem_list: bool = False,
    ):
        self.image_dir = Path(image_dir)
        self.num_samples_per_image = num_samples_per_image
        self.image_size = image_size
        self.object_detector = object_detector
        self.transform = transform
        self.return_stem_list = return_stem_list

        self.image_paths = self._collect_image_paths(self.image_dir)
        self.image_stems = [p.stem for p in self.image_paths]

        # 한 이미지에서 여러 trial/sample을 만들 수 있도록
        self.index_map = []
        for img_idx in range(len(self.image_paths)):
            for trial_idx in range(self.num_samples_per_image):
                self.index_map.append((img_idx, trial_idx))

    def _collect_image_paths(self, image_dir: Path) -> List[Path]:
        paths = []
        for p in sorted(image_dir.iterdir()):
            if not p.is_file():
                continue
            if p.name.startswith("."):
                continue
            if p.suffix.lower() not in VALID_IMAGE_EXTS:
                continue
            paths.append(p)
        return paths

    def get_image_stem_list(self) -> List[str]:
        return self.image_stems

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int):
        img_idx, trial_idx = self.index_map[index]
        image_path = self.image_paths[img_idx]
        image_stem = image_path.stem

        image = Image.open(image_path).convert("RGB")
        crop, bbox, sample_score = self._sample_object_crop(image, trial_idx)
        crop = self._resize_with_padding(crop, self.image_size)

        if self.transform is not None:
            crop = self.transform(crop)
        else:
            crop = self._pil_to_tensor(crop)

        sample = {
            "image": crop,
            "image_stem": image_stem,
            "trial_index": trial_idx,
            "bbox": torch.tensor(bbox, dtype=torch.float32),
            "sample_score": torch.tensor(sample_score, dtype=torch.float32),
        }

        if self.return_stem_list:
            sample["stem_list"] = self.image_stems

        return sample
    
    def _pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        arr = np.array(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2,0,1)
        return tensor

    def _sample_object_crop(self, image: Image.Image, trial_idx: int):
        """
        핵심:
        - detector가 있으면 detector 기반 bbox 후보 생성
        - 없으면 fallback으로 중앙/다중 비율 crop 후보 생성
        - 후보 중 점수가 높은 crop 선택
        """
        width, height = image.size

        candidate_boxes = self._generate_candidate_boxes(image, width, height, trial_idx)
        
        best_box = None
        best_score = -1e9

        for box in candidate_boxes:
            score = self._score_crop(image, box)
            if score > best_score:
                best_score = score
                best_box = box

        x1, y1, x2, y2 = self._expand_box(best_box, width, height, margin_ratio=0.10)
        crop = image.crop((x1, y1, x2, y2))
        return crop, (x1, y1, x2, y2), best_score
    
    def _expand_box(self, box, width: int, height: int, margin_ratio: float = 0.10):
        x1, y1, x2, y2 = box

        bw = x2 - x1
        bh = y2 - y1

        x1 = max(0, int(x1 - bw * margin_ratio))
        y1 = max(0, int(y1 - bh * margin_ratio))
        x2 = min(width, int(x2 + bw * margin_ratio))
        y2 = min(height, int(y2 + bh * margin_ratio))

        return (x1, y1, x2, y2)
    
    def _resize_with_padding(self, image: Image.Image, target_size=(512, 512), fill=(0, 0, 0)):
        target_w, target_h = target_size
        w, h = image.size

        scale = min(target_w / w, target_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        resized = image.resize((new_w, new_h))
        canvas = Image.new("RGB", (target_w, target_h), fill)

        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        canvas.paste(resized, (paste_x, paste_y))

        return canvas
    def _edge_penalty(self, box, width: int, height: int, edge_margin: int = 5) -> float:
        x1, y1, x2, y2 = box
        penalty = 0.0

        if x1 <= edge_margin:
            penalty += 1.0
        if y1 <= edge_margin:
            penalty += 1.0
        if x2 >= width - edge_margin:
            penalty += 1.0
        if y2 >= height - edge_margin:
            penalty += 1.0

        return penalty

    def _generate_candidate_boxes(self, image, width, height, trial_idx: int):
        candidates = []

        # 1) detector가 있으면 detector 결과 우선 사용
        if self.object_detector is not None:
            det_boxes = self.object_detector(image)
            for box in det_boxes:
                candidates.append(self._clamp_box(box, width, height))

        # 2) fallback: 중앙 crop + trial 기반 jitter crop
        candidates.append(self._center_box(width, height, scale=0.8))
        candidates.append(self._center_box(width, height, scale=0.6))
        candidates.append(self._center_box(width, height, scale=0.4))

        rng = random.Random(trial_idx + width * 31 + height * 17)
        for _ in range(6):
            scale = rng.uniform(0.45, 0.9)
            bw = int(width * scale)
            bh = int(height * scale)

            if bw < 8 or bh < 8:
                continue

            x1 = rng.randint(0, max(0, width - bw))
            y1 = rng.randint(0, max(0, height - bh))
            x2 = x1 + bw
            y2 = y1 + bh
            candidates.append((x1, y1, x2, y2))

        return candidates

    def _center_box(self, width: int, height: int, scale: float):
        bw = int(width * scale)
        bh = int(height * scale)
        x1 = max(0, (width - bw) // 2)
        y1 = max(0, (height - bh) // 2)
        x2 = min(width, x1 + bw)
        y2 = min(height, y1 + bh)
        return (x1, y1, x2, y2)

    def _clamp_box(self, box, width, height):
        x1, y1, x2, y2 = box
        x1 = max(0, min(int(x1), width - 1))
        y1 = max(0, min(int(y1), height - 1))
        x2 = max(x1 + 1, min(int(x2), width))
        y2 = max(y1 + 1, min(int(y2), height))
        return (x1, y1, x2, y2)

    def _score_crop(self, image: Image.Image, box) -> float:
        width, height = image.size
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1

        area_ratio = (bw * bh) / float(width * height + 1e-8)
        aspect = bw / float(bh + 1e-8)

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dx = abs(cx - width / 2.0) / max(width, 1)
        dy = abs(cy - height / 2.0) / max(height, 1)
        center_bonus = 1.0 - (dx + dy)

        # 사물이 너무 작거나 너무 큰 crop 방지
        target_area_ratio = 0.5
        size_score = -abs(area_ratio - target_area_ratio)

        # 지나치게 길쭉한 박스 패널티
        aspect_penalty = -abs(aspect - 1.0)

        # 화면 끝에 너무 붙은 박스 패널티
        edge_pen = self._edge_penalty(box, width, height, edge_margin=5)

        score = (
            2.0 * center_bonus
            + 1.5 * size_score
            + 0.5 * aspect_penalty
            - 1.2 * edge_pen
        )

        return float(score)