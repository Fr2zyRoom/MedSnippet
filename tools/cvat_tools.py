import os
import numpy as np
import cv2
from functools import reduce
import lxml.etree as ET

from tools.mask_tools import *

def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def cvat_wirte_polygon_xml(mask_dict, image_size, _id=0, filename="anonymized", xml_path=None):
    """
    write cvat annotation xml from mask
    
    Parameters:
        mask_dict (dict) -- description(str):mask(np.array)
        image_size (tuple) -- width, height
        description (string) -- mask description
        filename (string) -- file name
        xml_path (string) -- savepoint (../###.xml)

    
    """
    _annotations = ET.Element("annotations") 
    _version = ET.Element("version")
    _annotations.append(_version)
    ET.SubElement(_annotations, "version").text = "1.1"
    _image = ET.Element("image", 
                       id=str(_id),
                       name=filename,
                       width=str(image_size[0]),
                       height=str(image_size[1]))
    
    if mask_dict is not None:
        for description, mask in mask_dict.items():
            approx_contours = mask2poly(mask)
            cnt2str_ls = [";".join([str(w)+','+str(h) for w, h in np.squeeze(cnt)]) for cnt in approx_contours]
            for cnt2str in cnt2str_ls:
                _polygon = ET.Element("polygon", 
                                label=description, 
                                occluded="0", 
                                source="manual", 
                                points=cnt2str,
                                z_order="0")
                _image.append(_polygon)
        _annotations.append(_image)

    indent(_annotations)
    #ET.dump(_annotations)

    ET.ElementTree(_annotations).write(xml_path, encoding="utf-8")


def cvat_xml2mask(xml_path):
    """
    cvat annotation xml to mask
    
    Parameters:
        xml_path (dict) -- xml path

    Return:
        mask_dict (dict) -- description(str):mask(np.array)
    
    """
    # parse xml and get root
    tree = ET.parse(xml_path)
    root = tree.getroot()
    total_mask_dict = {}
    for _image in root.findall("./image"):
        # extract polygon info from xml
        fname = os.path.splitext(_image.attrib['name'])[0]
        w,h = int(_image.attrib['width']), int(_image.attrib['height'])
        _polygons = _image.findall("./polygon")
        poly_dict = {}
        for p in _polygons:
            p_dict = p.attrib
            description = p_dict['label']
            points = np.array([s.split(',') for s in p_dict['points'].split(';')]).astype(np.float)
            if poly_dict.get(description) is None:
                poly_dict[description] = [points]
            else:
                poly_dict[description].append(points)
        
        # get mask from polygon
#         mask_dict= {}
#         for lbl, poly in poly_dict.items():
#             mask = reduce(lambda a, b: a|b, [poly2mask((w,h), p) for p in poly])
#             mask_dict[lbl] = mask
        mask_dict= {lbl:poly2mask((w,h), poly) for lbl, poly in poly_dict.items()}
        total_mask_dict[fname] = mask_dict
    return total_mask_dict