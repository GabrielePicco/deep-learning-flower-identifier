import os
import json
from google_images_download import google_images_download


def download_flower_validation_set(number_of_images_for_category=10):
    """
    Download a validation test set for every flower category (downloading it from google images)
    :param number_of_images_for_category:
    :return:
    """
    response = google_images_download.googleimagesdownload()
    output_path = "./google_test_data"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open('cat_to_name.json', 'r') as f:
        number_to_categoryname = json.load(f)

    prefix = "flower"
    keywords = ["{} {}".format(prefix, x) for x in number_to_categoryname.values()]
    arguments = dict(keywords=",".join(keywords), limit=number_of_images_for_category,
                     output_directory=output_path, print_urls=True)
    response.download(arguments)

    categoryname_to_number = {v: k for k, v in number_to_categoryname.items()}

    for dir in os.listdir(output_path):
        try:
            dst = categoryname_to_number[dir[len(prefix) + 1:]]
            os.rename(os.path.join(output_path, dir), os.path.join(output_path, dst))
        except KeyError:
            pass


download_flower_validation_set()
