from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def collate_fn(x):
    return x[0]


class FaceSpace:
    def __init__(self):
        self.mtcnn = MTCNN(device=DEVICE)
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

    def detect_faces(self, data_set, data_loader):
        data_aligned = []
        data_names = []
        for x, y in data_loader:
            x_aligned, _ = self.mtcnn(x, return_prob=True)
            if x_aligned is not None:
                data_aligned.append(x_aligned)
                data_names.append(data_set.idx_to_class[y])
        return data_aligned, data_names

    def get_embeddings(self, root):
        data_set = datasets.ImageFolder(root)
        data_set.idx_to_class = {i: c for c, i in data_set.class_to_idx.items()}
        data_loader = DataLoader(data_set, collate_fn=collate_fn)
        data_aligned, data_names = self.detect_faces(data_set, data_loader)
        data_aligned = torch.stack(data_aligned).to(DEVICE)
        data_embeddings = self.resnet(data_aligned).detach().cpu()
        return data_embeddings, data_names
