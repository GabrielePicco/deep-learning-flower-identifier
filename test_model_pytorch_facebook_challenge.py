import os
import random
import sys
import urllib.request
import zipfile

import numpy as np
import torch
from airtable import airtable
from torchvision import datasets, transforms


def calc_accuracy(model, input_image_size, use_google_testset=False, testset_path=None, batch_size=32,
                  norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225]):
    """
    Calculate the mean accuracy of the model on the test test
    :param use_google_testset: If true use the testset derived from google image
    :param testset_path: If None, use a default testset (missing image from the Udacity dataset,
    downloaded from here: http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz)
    :param batch_size:
    :param model:
    :param input_image_size:
    :param norm_mean:
    :param norm_std:
    :return: the mean accuracy
    """
    if use_google_testset:
        testset_path = "./google_test_data"
        url = 'https://www.dropbox.com/s/3zmf1kq58o909rq/google_test_data.zip?dl=1'
        download_test_set(testset_path, url)
    if testset_path is None:
        testset_path = "./flower_data_orginal_test"
        url = 'https://www.dropbox.com/s/da6ye9genbsdzbq/flower_data_original_test.zip?dl=1'
        download_test_set(testset_path, url)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device=device)
    with torch.no_grad():
        batch_accuracy = []
        torch.manual_seed(33)
        torch.cuda.manual_seed(33)
        np.random.seed(33)
        random.seed(33)
        torch.backends.cudnn.deterministic = True
        datatransform = transforms.Compose([transforms.RandomRotation(45),
                                            transforms.Resize(input_image_size + 32),
                                            transforms.CenterCrop(input_image_size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(norm_mean, norm_std)])
        image_dataset = datasets.ImageFolder(testset_path, transform=datatransform)
        dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=_init_fn)
        for idx, (inputs, labels) in enumerate(dataloader):
            if device == 'cuda':
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model.forward(inputs)
            _, predicted = outputs.max(dim=1)
            equals = predicted == labels.data
            print("Batch accuracy (Size {}): {}".format(batch_size, equals.float().mean()))
            batch_accuracy.append(equals.float().mean().cpu().numpy())
        mean_acc = np.mean(batch_accuracy)
        print("Mean accuracy: {}".format(mean_acc))
    return mean_acc


def _init_fn(worker_id):
    """
    It makes determinations applied transforms
    :param worker_id:
    :return:
    """
    np.random.seed(77 + worker_id)


def download_test_set(default_path, url):
    """
    Download a testset containing approximately 10 images for every flower category.
    The images were download with the download_testset script and hosted on dropbox.
    :param default_path:
    :return:
    """
    if not os.path.exists(default_path):
        print("Downloading the dataset from: {}".format(url))
        tmp_zip_path = "./tmp.zip"
        urllib.request.urlretrieve(url, tmp_zip_path, download_progress)
        with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(default_path)
        os.remove(tmp_zip_path)


def download_progress(blocknum, blocksize, totalsize):
    """
    Show download progress
    :param blocknum:
    :param blocksize:
    :param totalsize:
    :return:
    """
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize: # near the end
            sys.stderr.write("\n")
    else: # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))


def publish_evaluated_model(model, input_image_size, username, model_name=None, optim=None, criteria=None,
                            scheduler=None, epoch=-1, comment=None, norm_mean=[0.485, 0.456, 0.406],
                            norm_std=[0.229, 0.224, 0.225], use_google_testset=False):
    """
    Publish the result on an airtable shared leaderboard
    :param model:
    :param input_image_size:
    :param username:
    :param model_name:
    :param optim:
    :param criteria:
    :param scheduler:
    :param epoch:
    :param comment:
    :param norm_mean:
    :param norm_std:
    :return:
    """
    at = airtable.Airtable('appQHMJgKMFqTjd9K', 'key9Wz1SXOE3UwuSd')
    if use_google_testset:
        table_name = "Leaderboard (Google Test Set)"
    else:
        table_name = "Leaderboard (Original Test Set)"
    mean_acc = calc_accuracy(model, input_image_size, norm_mean=norm_mean, norm_std=norm_std,
                             use_google_testset=use_google_testset)
    mean_acc = round(mean_acc, 7)
    records = at.get(table_name)['records']
    prec_id = 0
    prec_acc = 0
    override = False
    alredy_exist = False
    for entry in records:
        if entry['fields']['Username'] == username:
            alredy_exist = True
            prec_id = entry['id']
            prec_acc = entry['fields']['Accuracy']
            if float(prec_acc) < float(mean_acc):
                override = True
    record = {
        "Username": username,
        "Optim": optim,
        "Criteria": criteria,
        "Scheduler": scheduler,
        "Model": model_name if model_name is not None else type(model).__name__,
        "Epoch": None if epoch == -1 else str(epoch),
        "Accuracy": str(mean_acc),
        "Comments": comment
    }
    if override:
        at.update_all(table_name, prec_id, record)
        print("\nThe new model has exceeded the previous one: Accuracy on test set was {}, now is {}".format(prec_acc,
                                                                                                             mean_acc))
    elif not alredy_exist:
        at.create(table_name, record)
        print("\nYour model performance are now on Airtable: {}".format("https://airtable.com/shrCs1LDFdBus0tMG"))
    else:
        print("\nPrevious model was better: Accuracy on test set was {}, now is {}".format(prec_acc, mean_acc))
