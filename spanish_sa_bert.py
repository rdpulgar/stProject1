from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
import streamlit as st
import time

tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")  #sagorsarker/codeswitch-spaeng-sentiment-analysis-lince
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment") # "sagorsarker/codeswitch-spaeng-sentiment-analysis-lince"

nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

def main():

    st.title('Coke.ai')
    st.title('Análisis de sentimiento usanso BERT')

    # text = st.text_input("Expresión:")
    write_here = "Texto aqui..."
    text = st.text_area("Incluya un texto ..", write_here)
    if st.button("Analizar"):
        if text != write_here:
            label, score = sentimiento(text)
            st.success('Sentimiento de ['+ text + ']')
            st.success(label)
            st.success('%.2f' % score)    
        else:
            st.error("Ingresa un texto y presiona el boton Analizar ..")
    else:
        st.info(
            "Ingresa un texto y presiona el boton Analizar .."
        )

    uploaded_file = st.file_uploader("O bien puede seleccionar un archivo CSV para procesar múltiples párrafos (se procesará columna 'text')",type=['csv'])
    if uploaded_file is not None:
        if st.button("Procesar Archivo CSV"):
            data = pd.read_csv(uploaded_file,usecols=["text"])
            st.success("Procesando CSV ..")
            data[data['text'].str.strip().astype(bool)]
            #indexes = data[data.text_len <  30].index
            #data = data.drop(indexes)
            data['text'] = data['text'].astype(str)
            total_reg = len(data)
            #my_bar = st.progress(0)
            #i = 0
            t0 = time.time()
            msg = f"Espere por favor, esto puede tomar algun tiempo .. procesando {total_reg:.0f} elementos"  if total_reg>1000 else f"Espere .. procesando {total_reg:.0f} elementos"
            with st.spinner(msg):
                g = lambda x: pd.Series(sentimiento(x.text))
                data[['label', 'score']] = data.apply(g, axis=1)
            csv = convert_df(data)
            st.success(f'{total_reg:.0f} registros procesados con éxito en {time.time() - t0:.0f} seg')
            if st.download_button(label="Presione para descargar archivo procesado", data=csv, file_name='_cokeai_results.csv', mime='text/csv'):
                st.success("Descargado con éxito ..")
                st.stop()
        else:
            st.error("Aun no se ha procesado el archivo")
    else:
        st.info("Aun no se ha procesado el archivo")


def sentimiento(text):
    try:
        conditions = {
            1: 'Muy Malo',
            2: 'Malo',
            3: 'Neutro',
            4: 'Bueno',
            5: 'Muy bueno'
        }

        result = nlp(text)[0]
        x = int(result['label'][:1])
        if x in conditions.keys():
            return conditions[x], round(result['score'],4)
        else:
            return "_TextTooLong", -1
    except:
        return "_TextTooLong", -1

@st.cache
def convert_df(df):
    # Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


if __name__ == '__main__':
    main()