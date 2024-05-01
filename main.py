import streamlit as st
import numpy as np
import pandas as pd
import random
import os

from PIL import Image, ImageDraw
from PIL.ExifTags import TAGS, GPSTAGS
from ultralytics import YOLO

BOX_COLORS = [(0, 255, 0), (255, 0, 255)]

CSV_DATA_PATH = './data/'


def draw_boxes_on_image(img, results):
    # Goes through all results
    for result in results:
        draw = ImageDraw.Draw(img)
        all_box_conf = result.boxes.conf.tolist()
        all_box_types = result.boxes.cls.tolist()
        all_box_coords = result.boxes.xyxy.tolist()

        # Goes through all boxes in the current result
        for box_no in range(len(all_box_coords)):
            box_type = int(all_box_types[box_no])
            box_coords = all_box_coords[box_no]
            x0, x1 = map(int, [box_coords[0], box_coords[2]])
            y0, y1 = map(int, [box_coords[1], box_coords[3]])

            color = BOX_COLORS[box_type]

            # Confidence gradient from black (0) to fully saturated (1)
            color = tuple(map(lambda x : int(x * all_box_conf[box_no]), color))

            draw.rectangle(xy = (x0, y0, x1, y1),
                            fill = color)


def get_exif_data(img):
    exif_data = {}
    try:
        data = img._getexif()
        if data is None:
            st.write("Error: Image does NOT contain any EXIF metadata!")
            st.write("If you want to know where the photo was taken please \
                      supply an image that does contain GPS coordinates.")
            return exif_data
        for tag, value in data.items():
            decoded_tag = TAGS.get(tag, tag)
            if decoded_tag == "GPSInfo":
                gps_data = {}
                for gps_tag in value:
                    gps_decoded_tag = GPSTAGS.get(gps_tag, gps_tag)
                    gps_data[gps_decoded_tag] = value[gps_tag]
                exif_data[decoded_tag] = gps_data
            else:
                exif_data[decoded_tag] = value
    except (IOError, AttributeError, KeyError, IndexError) as err:
        print("Error: ", err)
    return exif_data


def get_gps_coords(exif_data):
    lat_deg, lat_min, lat_sec = exif_data["GPSInfo"]["GPSLatitude"]
    lat_dir = exif_data["GPSInfo"]["GPSLatitudeRef"]
    lon_deg, lon_min, lon_sec = exif_data["GPSInfo"]["GPSLongitude"]
    lon_dir = exif_data["GPSInfo"]["GPSLongitudeRef"]

    lat = (float(lat_deg) + lat_min / 60 + lat_sec / 3600)
    lon = (float(lon_deg) + lon_min / 60 + lon_sec / 3600)

    if lat_dir == "S":
        lat = -lat
    if lon_dir == "W":
        lon = -lon

    return [[lat, lon]]


def handle_uploaded_image(uploaded_file):
    display_image = st.button("Display the image")
    predict_image = st.button("Predict")
    map_prediction = st.button("Display position on map")
    save_gps_data = st.button("Save GPS data")
    st.button("Cancel")
    results = None

    exif_data = get_exif_data(Image.open(uploaded_file))
    gps_coords = get_gps_coords(exif_data)
    gps_data = pd.DataFrame(
        gps_coords,
        columns=["LAT", "LON"])

    if display_image:
        st.image(uploaded_file)

    if predict_image:
        if results is None:
            results = model.predict(source=Image.open(uploaded_file))
        modified_image = Image.open(uploaded_file)

        draw_boxes_on_image(modified_image, results)

        st.image(modified_image)

    if map_prediction:
        st.write(gps_data)
        st.map(gps_data)

    if save_gps_data:
        st.write("Data hopefully saved!")
        gps_data.to_csv(CSV_DATA_PATH + uploaded_file.name + ".csv")


def file_test():
    uploaded_file = st.file_uploader("This is the file uploader!!!")
    if uploaded_file is not None:
        handle_uploaded_image(uploaded_file)
    else:
        st.write("Please supply a file!")


def gather_all_coords():
    coords = pd.DataFrame()
    for f in [f for f in os.listdir(CSV_DATA_PATH) if os.path.isfile(CSV_DATA_PATH + f) and f[-4:] == ".csv"]:
        curr_coords = pd.read_csv(CSV_DATA_PATH + f)
        coords = pd.concat([coords, curr_coords])
    return coords


def data_test():
    uploaded_file = st.file_uploader("Please upload a CSV File!")

    if uploaded_file is not None:
        gps_data = pd.read_csv(uploaded_file)
        st.map(gps_data)
    else:
        st.write("OR")
        show_all_data = st.button("Show all available data on the server")
        if show_all_data:
            st.write("Found {} files!".format(len([f for f in os.listdir(CSV_DATA_PATH) if os.path.isfile(CSV_DATA_PATH + f) and f[-4:] == ".csv"])))
            all_coords = gather_all_coords()
            st.map(all_coords)


if not os.path.exists(CSV_DATA_PATH):
    os.mkdir(CSV_DATA_PATH)

model = YOLO("best.pt")

pages = {
    "File upload": file_test,
    "Show data": data_test
}

page_name = st.sidebar.selectbox("Choose a page", pages.keys())
pages[page_name]()
