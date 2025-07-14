import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class LSOTBTIRSubsetDataset(BaseDataset):
    """
    LSOTB-TIR test set.
    Publication:
        Large-scale Object Tracking in the Thermal Infrared Domain
        Qiao Liu, Xin Li, Zhenyu He, Nana Fan, Di Yuan, and Huchuan Lu
        IEEE Transactions on Image Processing (TIP), 2021
        https://arxiv.org/abs/2005.07829
    Download the dataset from https://github.com/QiaoLiuHit/LSOTB-TIR
    """
    def __init__(self):
        super().__init__()
        # LSOTB-TIR 데이터셋의 평가 데이터가 있는 경로를 설정합니다.
        self.base_path = self.env_settings.lsotb_tir_subset_path
        self.sequence_list = self._get_sequence_list()
        self.clean_list = self.clean_seq_list()

    def clean_seq_list(self):
        clean_lst = []
        for seq in self.sequence_list:
            # 시퀀스 이름에서 클래스를 추출합니다. (예: 'car_S_001' -> 'car')
            cls = seq.split('_')[0]
            clean_lst.append(cls)
        return clean_lst

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        # LSOTB-TIR은 LaSOT과 달리 클래스별 하위 폴더가 없습니다.
        anno_path = '{}/{}/groundtruth_rect.txt'.format(self.base_path, sequence_name)
        
        # groundtruth.txt 파일을 로드합니다. 구분자는 쉼표(,) 또는 공백일 수 있습니다.
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64, backend='numpy')

        # LSOTB-TIR은 별도의 occlusion/out_of_view 파일이 없습니다.
        # 대신, ground truth의 width와 height가 0보다 큰지를 기준으로 객체 가시성을 판단합니다.
        target_visible = np.logical_and(ground_truth_rect[:, 2] > 0, ground_truth_rect[:, 3] > 0)
        
        # 이미지 프레임 경로를 생성합니다.
        frames_path = '{}/{}/img'.format(self.base_path, sequence_name)
        
        # 이미지 파일명은 4자리 숫자(e.g., 0001.jpg)로 되어 있습니다.
        frames_list = ['{}/{:04d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]

        target_class = sequence_name.split('_')[0]
        
        return Sequence(sequence_name, frames_list, 'lsotb_tir', ground_truth_rect.reshape(-1, 4),
                        object_class=target_class, target_visible=target_visible)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        # LSOTB-TIR 데이터셋의 시퀀스 목록 (예시)
        # 실제 사용 시에는 전체 시퀀스 목록으로 대체해야 합니다.
        sequence_list = ['airplane_H_001',
                         ] # ... and so on for all sequences
        return sequence_list