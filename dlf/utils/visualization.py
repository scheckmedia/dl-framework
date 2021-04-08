import numpy as np
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from dlf.utils.helpers import cmap
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')

flat_cmap = cmap.flatten()

font_name = "DejaVuSans-Bold.ttf"
try:
    font = ImageFont.truetype(font_name, 14)
except:
    font = ImageFont.load_default()
    tf.get_logger().warn('Font "{}" not found use default font'.format(font_name))


def visualize_object_detection(batch_images, batch_boxes, batch_labels, batch_scores, batch_gt_boxes, batch_gt_labels, classes=None):
    shape = batch_images.shape * np.array([1, 1, 2, 1])
    output = np.zeros((shape), dtype=np.uint8)

    for i in range(0, batch_images.shape[0]):
        src_image = batch_images[i]
        boxes = batch_boxes[i]
        scores = batch_scores[i]
        labels = batch_labels[i]
        gt_boxes = batch_gt_boxes[i]
        gt_labels = batch_gt_labels[i]
        sizes = np.repeat(src_image.shape[:-1], 2)
        label_y_offset = 4

        if boxes.shape[0] > 0:
            pil_image = src_image.numpy().squeeze().astype(np.uint8)
            pil_image = Image.fromarray(pil_image)
            draw = ImageDraw.Draw(pil_image)

            for box, score, label in zip(boxes, scores, labels):
                y1, x1, y2, x2 = (box.numpy() * sizes).astype(np.int32)
                if classes:
                    class_name = classes[int(label)]
                else:
                    class_name = "Class {}".format(label)

                text = "{}: {:.2f}".format(class_name, score)
                text_size = font.getsize(text)
                draw.text((x1, y1 - text_size[1] - label_y_offset), text, font=font, fill=tuple(cmap[label + 1]))
                draw.rectangle([x1, y1, x2, y2], outline=tuple(cmap[label + 1]))

            image = tf.squeeze(np.array(pil_image))

        if gt_boxes.shape[0] > 0:
            pil_image = src_image.numpy().squeeze().astype(np.uint8)
            pil_image = Image.fromarray(pil_image)
            draw = ImageDraw.Draw(pil_image)

            for box, label in zip(gt_boxes, gt_labels):
                y1, x1, y2, x2 = (box * sizes).astype(np.int32)
                if classes:
                    class_name = classes[int(label)]
                else:
                    class_name = "Class {}".format(label)

                text_size = font.getsize(class_name)
                draw.text((x1, y1 - text_size[1] - label_y_offset), class_name, font=font, fill=tuple(cmap[label + 1]))
                draw.rectangle([x1, y1, x2, y2], outline=tuple(cmap[label + 1]))

            gt_image = tf.squeeze(np.array(pil_image))
            output[i, :, :] = np.concatenate(
                (image, gt_image), axis=1)

    return output


def visualize_predictions_only(image, pred, opacity=0.6):
    pred = pred.astype(np.uint8)
    image = ((image.astype(np.float32) / 2.0) + 0.5) * 255.0
    output = np.zeros((image.shape), dtype=np.uint8)

    for i in range(0, image.shape[0]):
        batch_img = image[i, :, :, :]
        batch_pred = pred[i, :, :]

        pred_img = Image.fromarray(batch_pred, mode='P')
        pred_img.putpalette(flat_cmap)
        pred_img = np.array(pred_img.convert(
            'RGB')).astype(np.float32) * opacity
        pred_img = np.clip(
            pred_img + (1 - opacity) * batch_img, 0, 255).astype(np.uint8)

        output[i, :, :, :] = pred_img

    return output


def visualize_predictions(image, gt, pred, opacity=0.6):
    gt = gt.astype(np.uint8)
    pred = pred.astype(np.uint8)
    shape = image.shape * np.array([1, 1, 3, 1])
    output = np.zeros((shape), dtype=np.uint8)

    for i in range(0, image.shape[0]):
        batch_img = image[i, :, :, :]
        batch_pred = pred[i, :, :]
        batch_gt = gt[i, :, :]

        pred_img = Image.fromarray(batch_pred, mode='P')
        gt_img = Image.fromarray(batch_gt, mode='P')

        pred_img.putpalette(flat_cmap)
        gt_img.putpalette(flat_cmap)

        pred_img = np.array(pred_img.convert(
            'RGB')).astype(np.float32) * opacity
        gt_img = np.array(gt_img.convert('RGB')).astype(np.float32) * opacity

        pred_img = np.clip(
            pred_img + (1 - opacity) * batch_img, 0, 255).astype(np.uint8)
        gt_img = np.clip(
            gt_img + (1 - opacity) * batch_img, 0, 255).astype(np.uint8)

        output[i, :, :, :] = np.concatenate(
            (batch_img, gt_img, pred_img), axis=1)

    return output


def visualize_predictions_string(image, gt, pred, opacity=0.6):
    vis = visualize_predictions(image, gt, pred, opacity)
    buffers = []

    for i in range(vis.shape[0]):
        vis_image = Image.fromarray(vis[i, :, :, :])
        with BytesIO() as data:
            vis_image.save(data, format='PNG')
            buffers += [data.getvalue()]

    return buffers


def visualize_predictions_summary(image, gt, pred, opacity=0.6, prefix=''):
    summary_list = []
    data = visualize_predictions_string(image, gt, pred, opacity)

    if prefix != '':
        prefix += '/'

    for i in range(image.shape[0]):
        if len(data) != image.shape[0]:
            continue

        summary = tf.Summary.Image(
            encoded_image_string=data[i],
            height=image.shape[1],
            width=image.shape[2]
        )

        summary_list += [
            tf.Summary.Value(
                tag=prefix + "predictions/%s" % i,
                image=summary
            )
        ]

    return tf.Summary(value=summary_list)


def visualize_confusion_matrix(cm,
                               classes,
                               title='Confusion matrix',
                               cmap=None,
                               skip_first=False,
                               normalize=True):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if skip_first:
        cm = cm[1:, 1:]

    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else '.0f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    img = Image.frombytes('RGB',
                          canvas.get_width_height(),
                          canvas.tostring_rgb())

    buffer = BytesIO()
    img.save(buffer, 'PNG')
    plt.close()

    return np.array(img)


def visualize_cycle_gan(imgs, titles):
    fig, axis = plt.subplots(2, 3, figsize=(15, 15))
    cnt = 0
    for i in range(2):
        for j in range(3):
            if cnt == 0 or cnt == 3:
                img = imgs[cnt].astype(np.uint8)
            else:
                img = ((imgs[cnt] + 1) * 127.5).astype(np.uint8)

            axis[i, j].imshow(img)
            axis[i, j].set_title(titles[j])
            axis[i, j].axis('off')
            cnt += 1

    fig.tight_layout()
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    img = Image.frombytes('RGB',
                          canvas.get_width_height(),
                          canvas.tostring_rgb())

    buffer = BytesIO()
    img.save(buffer, 'PNG')
    plt.close()

    return np.array(img)
