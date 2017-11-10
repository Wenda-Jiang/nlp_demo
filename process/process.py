# coding=utf-8
from __future__ import print_function
import tensorflow as tf
import json

def log_tensor_shap(tensor, sess):
  print(tensor)
  print(sess.run(tf.shape(tensor)))


def process_caption(file):

  with tf.gfile.GFile(file, 'r') as f:
    caption_data = json.load(f)
    id_to_filename = dict([(x["id"], x["file_name"]) for x in caption_data["images"]])

    # Extract the captions. Each image_id is associated with multiple captions.
    id_to_captions = {}
    for annotation in caption_data["annotations"]:
      image_id = annotation["image_id"]
      caption = annotation["caption"]
      id_to_captions.setdefault(image_id, [])
      id_to_captions[image_id].append(caption)

    return id_to_filename, id_to_captions

def process_image_file(file):
  """
  read file, decode as tensor
  :param file: 
  :return: 
  """
  with tf.gfile.GFile(file, 'r') as f:
    image = f.read()

    # 3_D tensor [height, width, channels]
    g = tf.Graph()

    with g.as_default():
      # decode image
      image = tf.image.decode_jpeg(image, channels=3)

    with tf.Session(graph=g) as sess:
      # Tensor("DecodeJpeg:0", shape=(?, ?, 3), dtype=uint8)  shape: [640 591 3]
      log_tensor_shap(image, sess)

      image = tf.image.resize_images(image,
                                     size=[346, 346],
                                     method=tf.image.ResizeMethod.BILINEAR)

      # Tensor("Squeeze:0", shape=(346, 346, 3), dtype=float32) shape: [346 346   3]
      log_tensor_shap(image, sess)

      image = tf.image.resize_image_with_crop_or_pad(image, 269, 269)

      # Tensor("Squeeze_1:0", shape=(269, 269, 3), dtype=float32) shape: [269 269   3]
      log_tensor_shap(image, sess)

      # elements 0／1 to -1/1
      image = tf.subtract(image, 0.5)
      image = tf.multiply(image, 2.0)

      return image


if __name__ == '__main__':
  image = process_image_file('../data/train2014/COCO_train2014_000000000025.jpg')
  id_to_filename, id_to_captions = process_caption('../data/annotations/captions_train2014.json')
  print(id_to_filename[25], id_to_captions[25])
