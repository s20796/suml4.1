# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pickle
from datetime import datetime
startTime = datetime.now()
# import znanych nam bibliotek

import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

filename = "model.sv"
model = pickle.load(open(filename,'rb'))
# otwieramy wcześniej wytrenowany model

def main():

	st.set_page_config(page_title="Diagnoza App")
	overview = st.container()
	left, right = st.columns(2)
	prediction = st.container()

	st.image("https://ocdn.eu/images/pulscms/ZDk7MDA_/362775ff399ba0e474fdf4e6b57feddf.jpeg")

	with overview:
		st.title("Diagnoza App")

	with right:
		objawy_slider = st.slider("Objawy", value=1, min_value=1, max_value=5)
		choroby_wsp_slider = st.slider("Wiek", min_value=11, max_value=77)
		wzrost_slider = st.slider("Wzrost", min_value=159, max_value=200)
		leki_slider = st.slider("Leki", min_value=1, max_value=4)

	data = [[objawy_slider, choroby_wsp_slider, wzrost_slider, leki_slider]]
	survival = model.predict(data)
	s_confidence = model.predict_proba(data)

	with prediction:
		st.subheader("Czy taka osoba jest chora?")
		st.subheader(("Tak" if survival[0] == 1 else "Nie"))
		st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))

if __name__ == "__main__":
    main()
