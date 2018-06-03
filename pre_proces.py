import glob, os, csv
from PIL import Image, ImageOps, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

SIZE = 128
directory = './data/original-data'
image_dir = './data/images'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

image_id = 1
class_list = []
with open('./data/train.csv', 'w') as train:
    with open('./data/test.csv', 'w') as test:
        train_writer = csv.writer(train, delimiter=',')
        test_writer = csv.writer(test, delimiter=',')

        author_class = 0
        for item in os.listdir(directory):
            path = os.path.join(directory, item)
            if os.path.isdir(path):
                os.chdir(path)
                images = glob.glob("*.jpg")
                size = len(images)

                image_count = 0
                # re-scale image and move to /data/images
                for src in images:
                    image_name = '{}.jpg'.format(image_id)
                    # image = Image.open(src).convert('RGB')
                    # image = ImageOps.fit(image, (SIZE, SIZE), Image.ANTIALIAS, 0, (0.5, 0.5))
                    # image.save(os.path.join('../../images', image_name))
                    image_id += 1
                    image_count += 1

                    # write sort out train/test set
                    if image_count < size * 0.8:
                        train_writer.writerow([image_name, author_class])
                    else:
                        test_writer.writerow([image_name, author_class])

                os.chdir('../../../')
                class_list.append((item, author_class))
                author_class += 1

with open('./data/class-labels.txt', 'w') as f:
    for (name, label) in class_list:
        f.write("{}, {}\n".format(name, label))