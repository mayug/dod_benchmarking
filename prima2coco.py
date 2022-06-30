import os
import argparse
import json
import xml.etree.ElementTree as ET
from typing import Dict, List
from tqdm import tqdm
import re
import numpy as np

# Update
# added 'img' variable that contains filenames to function get_image_info() because the xml annotations in CDDOD don't contain filenames.
# added polygon segmentations (with 4 vertices) to compare with output of DocSegTr segmentations

def get_label2id(labels_path: str) -> Dict[str, int]:
    """id is 1 start"""
    with open(labels_path, 'r') as f:
        labels_str = f.read().split()
    labels_ids = list(range(1, len(labels_str)+1))
    return dict(zip(labels_str, labels_ids))


def get_annpaths(ann_dir_path: str = None,
                 ann_ids_path: str = None,
                 ext: str = '',
                 annpaths_list_path: str = None) -> List[str]:
    # If use annotation paths list
    if annpaths_list_path is not None:
        with open(annpaths_list_path, 'r') as f:
            ann_paths = f.read().split()
        return ann_paths

    # If use annotaion ids list
    ext_with_dot = '.' + ext if ext != '' else ''
    with open(ann_ids_path, 'r') as f:
        ann_ids = f.read().split()
    ann_paths = [os.path.join(ann_dir_path, aid+ext_with_dot) for aid in ann_ids]
    return ann_paths


def get_image_info(img, page, extract_num_from_imgid=True):


    img_name = img

    img_id = os.path.splitext(img_name)[0]

    # print([img_name, img_id])

    
    if extract_num_from_imgid and isinstance(img_id, str):
        img_id = int(img_id)

    # print(img_id)
    # asd
    size = page.attrib

    width = int(size['imageWidth'])
    height = int(size['imageHeight'])

    image_info = {
        'file_name': img, #filename,
        'height': height,
        'width': width,
        'id': img_id
    }
    return image_info


def get_coco_annotation_from_obj(obj, label2id):
    label = obj.tag.split('}')[-1]
    # print('label ', obj.attrib)
    assert label in label2id, f"Error: {label} is not in label2id !"
    category_id = label2id[label]
    coords = _find(obj, "Coords")
    points = list(coords)

    x_list = []
    y_list = []
    for p in points:
        x_list.append(int(p.attrib['x']))
        y_list.append(int(p.attrib['y']))
    
    if len(x_list)==0 or len(y_list)==0:
        return 'no_coords'

    xmin, xmax = min(x_list)-1, max(x_list)
    ymin, ymax = min(y_list)-1, max(y_list)
    seg_polygon= [(x,y) for x,y in zip(x_list, y_list)]
    seg_polygon = [int(i) for i in list(np.ravel(seg_polygon))]

    if len(seg_polygon) == 4:
            seg_polygon = [[xmin, ymin]]
            seg_polygon[0].insert(len(seg_polygon[0]), xmax)
            seg_polygon[0].insert(len(seg_polygon[0]), ymin)
            
            seg_polygon[0].insert(len(seg_polygon[0]), xmax)
            seg_polygon[0].insert(len(seg_polygon[0]), ymax)
            
            seg_polygon[0].insert(len(seg_polygon[0]), xmin)
            seg_polygon[0].insert(len(seg_polygon[0]), ymax)
    else:
            seg_polygon = [seg_polygon]

    assert np.array(seg_polygon).ndim==2
    
    assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin


    
    ann = {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'ignore': 0,
        'segmentation': seg_polygon #added to compare with DocSegTr output #[]  # This script is not for segmentation
    }
    return ann


def _get_img_name(a_base):
    if 'pc' in a_base:
        a_base = a_base[len('pc-'):]
    a_base = a_base[:-len('.xml')]
    return a_base+'.png'


def _find(element, objecttag):
    possible_schema = ["{http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-01-12}",
                       "{http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19}",
                       "{http://schema.primaresearch.org/PAGE/gts/pagecontent/2009-03-16}"]


    for p in possible_schema:
        
        tag = p + objecttag
        # print('tag ', tag)
        found = element.find(tag)
        # print('found ', found)
        if found is None:
            continue
        else:
            return found



def convert_xmls_to_cocojson(annotation_paths: List[str],
                             label2id: Dict[str, int],
                             output_jsonpath: str,
                             extract_num_from_imgid: bool = True):
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    bnd_id = 1  # START_BOUNDING_BOX_ID, TODO input as args ?
    print('Start converting !')
    # print(annotation_paths)
    ctr = 0
    print('len(annotation_paths)', len(annotation_paths))
    for a_path in tqdm(annotation_paths):
        # Read annotation xml

        
        ann_tree = ET.parse(a_path)
        ann_root = ann_tree.getroot()
        

        a_base = os.path.basename(a_path)
        img = _get_img_name(a_base)
         
        # asd

        print('a_path ', a_path)

        page = _find(ann_root, "Page")
        objects = list(page)
    
        img_info = get_image_info(img, page=page,
                                  extract_num_from_imgid=extract_num_from_imgid)
        img_id = img_info['id']
        output_json_dict['images'].append(img_info)

        for obj in objects:
            ann = get_coco_annotation_from_obj(obj=obj, label2id=label2id)
            if ann == 'no_coords':
                continue
            ann.update({'image_id': img_id, 'id': bnd_id})
            output_json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for label, label_id in label2id.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)

    with open(output_jsonpath, 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)


def main():
    parser = argparse.ArgumentParser(
        description='This script support converting prima format xmls to coco format json')
    parser.add_argument('--ann_dir', type=str, default=None,
                        help='path to annotation files directory. It is not need when use --ann_paths_list')
    parser.add_argument('--ann_ids', type=str, default=None,
                        help='path to annotation files ids list. It is not need when use --ann_paths_list')
    parser.add_argument('--ann_paths_list', type=str, default=None,
                        help='path of annotation paths list. It is not need when use --ann_dir and --ann_ids')
    parser.add_argument('--labels', type=str, default=None,
                        help='path to label list.')
    parser.add_argument('--output', type=str, default='output.json', help='path to output json file')
    parser.add_argument('--ext', type=str, default='', help='additional extension of annotation file')
    args = parser.parse_args()

    label2id = get_label2id(labels_path=args.labels)

    print(label2id)
    

    ann_paths = get_annpaths(
        ann_dir_path=args.ann_dir,
        ann_ids_path=args.ann_ids,
        ext=args.ext,
        annpaths_list_path=args.ann_paths_list
    )
    convert_xmls_to_cocojson(
        annotation_paths=ann_paths,
        label2id=label2id,
        output_jsonpath=args.output,
        extract_num_from_imgid=True
    )


if __name__ == '__main__':
    main()
