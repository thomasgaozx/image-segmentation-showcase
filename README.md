# image-segmentation-showcase

The jupyter notebook from [Waterloo.AI Data Challenge](https://uwaterloo.ca/engineering/events/waterlooai-data-challenge-fall-2022) hackathon.
(Warning: code is  ugly due to time constraints)

Employed semantic segmentation to identify individual works against complex, moving factory backgrounds.
Firstly, manually label a few images using Microsoft Paint 🤮 (labeled 8 out of 13500 images).

Secondly, replace the last layer of a pre-trained resnet50 with a single layer of Conv2d with 2 classifications (background and person)

```python
model = deeplabv3_resnet50(pretrained=True)
model.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1,1), stride=(1,1))
```

Thirdly, fix the pre-trained weights, and only train the last layer with a high learning rate (1e-4).

```python
optimizer = torch.optim.Adam(model.classifier[4].parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()
## (…) standard training loop
```

Fourthly, train the entire model (including pre-trained weight) with a much lower learning rate (1e-5).
Sees dramatic improvement.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()
## (…) standard training loop
```

Results demonstration:

![image](https://github.com/thomasgaozx/image-segmentation-showcase/assets/29295494/c025dc57-3405-4ad8-84e0-912abe14e4a2)



