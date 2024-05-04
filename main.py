import streamlit as st
#from streamlit_image_comparison import image_comparison
import numpy as np
import pandas as pd
import random
import os

from PIL import Image, ImageDraw
from PIL.ExifTags import TAGS, GPSTAGS
from ultralytics import YOLO

BOX_COLORS = [(0, 255, 0), (255, 0, 255)]

CSV_DATA_PATH = './data/'
V9C_METRICS_PATH = './metrics/v9c/'
V8S_METRICS_PATH = './metrics/v8s/'
SAVE_DATA_TO_DISK = True


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


def get_potholes_and_manholes(results):
    ph = 0
    mh = 0
    for result in results:
        for box in result.boxes.cls.tolist():
            if int(box) == 0:
                mh += 1
            else:
                ph += 1
    return ph, mh


def get_exif_data(img):
    exif_data = {}
    try:
        data = img._getexif()
        if data is None:
            st.write("Warning: Image does NOT contain any EXIF metadata!")
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
        st.write("Error: ", err)
        print("Error: ", err)
    return exif_data


def get_gps_coords(exif_data):
    if "GPSInfo" not in exif_data.keys():
        return None

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
    model_choice = st.radio("Pick the model to use",
                            ["v8s", "v9c"])

    st.write(f"Using the YOLO ***{model_choice}*** model")

    if model_choice == "v8s":
        model = YOLO("bestv8s.pt")
    else:
        model = YOLO("bestv9c.pt")

    assert(model != None)

    show_metrics = st.button("Show selected model's metrics")
    display_image = st.button("Display the image")
    predict_image = st.button("Predict and save data")
    map_prediction = st.button("Display position on map")
    st.button("Cancel")
    results = None

    exif_data = get_exif_data(Image.open(uploaded_file))
    gps_coords = get_gps_coords(exif_data)
    if gps_coords is not None:
        gps_data = pd.DataFrame(
            gps_coords,
            columns=["LAT", "LON"])
    else:
        gps_data = None

    if show_metrics:
        if model_choice == "v8s":
            METRICS_PATH = V8S_METRICS_PATH
        else:
            METRICS_PATH = V9C_METRICS_PATH
        st.image(METRICS_PATH + "confusion_matrix.png")
        st.image(METRICS_PATH + "P_curve.png")
        st.image(METRICS_PATH + "results.png")

    if display_image:
        st.image(uploaded_file)

    if predict_image:
        if results is None:
            results = model.predict(source=Image.open(uploaded_file))
        modified_image = Image.open(uploaded_file)

        draw_boxes_on_image(modified_image, results)
        ph, mh = get_potholes_and_manholes(results)
        st.write(f"Potholes: {ph}")
        st.write(f"Manholes: {mh}")

        st.write("Drag the slider to see the image with/without bounding boxes")
        #image_comparison(img1=Image.open(uploaded_file), img2=modified_image)
        st.image(uploaded_file, caption="Before")
        st.image(modified_image, caption="After")

        if gps_data is None:
            st.write("No GPS data in image!")
            st.write("Nothing was saved!")
        else:
            gps_data["CNT"] = ph
            st.write("The resulting CSV file contains:")
            st.write(gps_data)
            if SAVE_DATA_TO_DISK:
                gps_data.to_csv(CSV_DATA_PATH + uploaded_file.name + ".csv", index=False)
                st.write("Data hopefully saved to disk!")
            st.download_button(
                label="Download the data",
                data=gps_data.to_csv(index=False),
                file_name="data.csv"
            )

    if map_prediction:
        if gps_data is None:
            st.write("No GPS data in image!")
        else:
            st.write(gps_data)
            st.map(gps_data)


def upload_data():
    uploaded_file = st.file_uploader("Please upload an image or video file!")
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


def view_data():
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


model = None
if not os.path.exists(CSV_DATA_PATH):
    os.mkdir(CSV_DATA_PATH)

pages = {
    "Process an image": upload_data,
    "View CSV file": view_data
}

page_name = st.sidebar.selectbox("Choose a page", pages.keys())
pages[page_name]()
