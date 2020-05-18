from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def collate_fn(x):
    return x[0]


class FaceSpace:
    def __init__(self, device=DEVICE, batch_size=32):
        self.device = device
        self.mtcnn = MTCNN(device=self.device)
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.batch_size = batch_size

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
        data_aligned = torch.stack(data_aligned).to(self.device)
        batches = []
        for batch in data_aligned.split(self.batch_size, dim=0):
            with torch.no_grad():
                batches.append(self.resnet(batch))
        return torch.cat(batches, dim=0).cpu(), data_names

    def detect_faces_from_images(self, images):
        x_aligned, _ = self.mtcnn(images, return_prob=True)
        return x_aligned

    def get_embeddings_from_images(self, images):
        data_aligned = torch.stack(self.detect_faces_from_images(images)).to(self.device)
        with torch.no_grad():
            embeddings = self.resnet(data_aligned).cpu()
        return embeddings
