# Face Recognition

This library provides out-of-the box facial recognition training and inference based on [Facenet](https://github.com/timesler/facenet-pytorch) and [Pytorch](https://pytorch.org/)

## Important Requirements

```
python >= 3.6
torch >=1.0.0
```

## Quick start
Install with pip.

```bash
pip install pl-face-recognition
```

This project was designed to minimize the amount of false positives, for that
it requires two sets of data, one with pictures of the people you want to recognize,
and other with examples of other random people (a public dataset like [lfw](http://vis-www.cs.umass.edu/lfw/) should suffice).

So given a dataset structure like this (we follow the [torchvision ImageFolder](https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.ImageFolder) structure, for both sets of data separately):

```
dataset/
__good_faces/
____hugh_jackman/
______hugh_1.jpeg
____ryan_reynolds/
______ryan_1.jpeg
__bad_faces/
____some_random_person/
______random_person_1.jpeg
____some_random_other_person/
______dude.jpeg
```

You may do insta train and inference like this:

```python
from face_recognition.face_recognizers import OneNeighborRecognizer
from PIL import Image

IMAGES_PATH = './dataset/good_faces'
ADVERSARY_IMAGES_PATH = './dataset/bad_faces'
TEST_IMAGE_PATH = './ryan_2.jpeg'


clf = OneNeighborRecognizer() # finds a match via 1 closest neighbor
clf.fit(IMAGES_PATH, ADVERSARY_IMAGES_PATH)
test = [Image.open(TEST_IMAGE_PATH)] # predict receives an array/batch of images

predicted_id = clf.predict(test)

print(predicted_id)
print(clf.ids_to_class(predicted_id))

>> 1
>> 'ryan_reynolds'
```

## Custom behavior
The library works by doing nearest neighbor search on FaceNet generated
embeddings of the known images. If you want to have direct access to the embeddings
you can do so via the FaceSpace class:

```python
from face_recognition.face_space import FaceSpace

IMAGES_PATH = './dataset/good_faces'
TEST_IMAGE_PATH = './ryan_2.jpeg'
test = [Image.open(TEST_IMAGE_PATH)]

fs = FaceSpace()
embeddings, class_names = fs.get_embeddings(IMAGES_PATH) # get embeddings from a dataset

image_embeddings = fs.get_embedding_from_images(test) # or from array of images
```
The library will provide different methods for using the embeddings for classification.
For know it only has the `OneNeighborClassifier` class:

```python
from face_recognition.classifiers import OneNeighborClassifier
classifier = OneNeighborClassifier()
classifier.fit(embeddings, labels) # labels are integer indexes for target classes and -1 for adversarial
classifier.predict(other_embeddings)
```
You may implement your own distance based classifiers by inheriting from `classifiers.DistanceClassifier` and defining the predict method.

```python
from face_recognition.classifiers import DistanceClassifier
import random

class StupidClassifier(DistanceClassifier):

    def predict(self, target_embeddings):
        '''
        here you define how to get the correct labels by somehow comparing
        self.embeddings and target_embeddings
        See OneNeighborClassifier for an example
        '''
        return  [random.choice(self.labels) for _ in range(len(target_embeddings))] # and you return labels
```

## Contributing

If you want to add functionality, fork the repo and feel free to open a pull request.

## Credits

Thank you [contributors](https://github.com/platanus/face-recognition/graphs/contributors)!

<img src="http://platan.us/gravatar_with_text.png" alt="Platanus" width="250"/>

face-recognition is maintained by [platanus](http://platan.us).

## License

face-recognition is © 2020 platanus, spa. It is free software and may be redistributed under the terms specified in the LICENSE file.