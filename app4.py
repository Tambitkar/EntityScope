import spacy_streamlit
import streamlit as st
import spacy
import pandas as pd
from pathlib import Path
import srsly
import importlib
import random
from spacy.pipeline import EntityRuler  # Import the Entity Ruler for making custom entities
from st_aggrid import AgGrid

st.set_page_config(layout="wide")

MODELS = srsly.read_json(Path(__file__).parent / "models.json")
DEFAULT_MODEL = "en_core_web_sm"
DEFAULT_TEXT = "Chitons collected by Dr. Harold Heath at Pacific Grove, near Monterey, California by H. A. Pilsbry. During the summer of 1897 Dr. Heath collected a series of invertebrates and fishes for the Academy, including some twenty-four species and varieties of Polyplacophora. At Pacific Grove, the typical Mopalia muscosa, typical Mopalia lignosa and typical Mopalia hindsii occur, without, so far as the series seen shows, any specimens of intermediate character. A Mopalia described below, differs from those hitherto known in the perfectly regular form of the tail valve, which is like that of Ischnochiton, thus breaking down, in large measure, the distinction between the Ischnochitonidae and the Mopaliidae. This lawless species is new."

DESCRIPTION = """"""
FOOTER = """<span style="font-size: 0.75em">&hearts; Built with [`spacy-streamlit`]</span>"""

@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')

# NOTE: custom patterns have already been created for the NLP Pipeline > Entity Ruler via the ruler.py file.

st.title("NER pipeline")

st.markdown("")

# SIDEBAR START ---------------------------- 
with st.sidebar:
    col1, col2, col3 = st.columns([1, 6, 1])

    with col1:
        st.write("")

    with col2:
        st.image("C:\\Users\\Rohit Tambitkar\\OneDrive\\Desktop\\syllabus\\NLP\\streamlit-ansp-master\\images\\pngwing.com.png")

    with col3:
        st.write("")

    st.write(DESCRIPTION)    
    st.write("")  # vertical spacing
    st.markdown("## How does it work?")
    st.markdown("Upload a text file or paste text into the box to explore named entities and the results of spaCy's natural language processing pipeline.")
    st.write("")  # vertical spacing
    st.markdown("")
    st.write("")
    st.write("")

    st.markdown("")
    st.write("")  # vertical padding
    st.markdown(FOOTER, unsafe_allow_html=True)

# SIDEBAR END ---------------------------- 

# FILE UPLOADER ---------------------------- 
st.markdown(":sparkles: **Upload Text File** :sparkles: ")
uploaded_file = st.file_uploader("File Upload", type=["txt"])
# FILE UPLOADER ----------------------------

st.markdown("---")

if uploaded_file is not None:
    
    nlp = spacy.load(DEFAULT_MODEL)
    ruler = nlp.add_pipe("entity_ruler", before='ner')
    ruler.from_disk(Path(__file__).parent / "../data/ansp-clean-patterns.jsonl")
    doc = nlp(uploaded_file.getvalue().decode("utf-8"))
    
    # to plot the labels for the custom entities, we have to make a new set
    labels = list(nlp.get_pipe("ner").labels)
    for label in nlp.get_pipe("entity_ruler").labels:
        labels.append(label)
        
    colors = {"CARDINAL": "#DB9D85", "DATE": "#D0A374", "EVENT": "#C2A968", "FAC": "#B1AF64", "GPE": "#9DB469",
              "HABITAT": "#86B875", "LANGUAGE": "#6DBC86", "LAW": "#53BE98", "LOC": "#3DBEAB", "MONEY": "#39BDBC",
              "NORP": "#4CB9CC", "ORDINAL": "#69B4D8", "ORG": "#87AEDF", "PERCENT": "#A3A7E2", "PERSON": "#BB9FE0",
              "PRODUCT": "#CD99D8", "QUANTITY": "#DA95CC", "TAXA": "#E293BD", "TIME": "#E494AB", "WORK_OF_ART": "#E29898"}

    spacy_streamlit.visualize_ner(
        doc,
        labels=labels,
        colors=colors,
        title="Custom Entity Labels",
        show_table=False,
    )

    st.markdown('## Exploring the Tokens')
    # Create dataframe for data download
    rows = []
    for token in doc:
        rows.append(
            {
                'Token': token.text,
                'Lemma': token.lemma_,
                'POS': token.pos_,
                'Tag': token.tag_,
                'Dependency': token.dep_,
                'Head': token.head.text if hasattr(token.head, 'text') else str(token.head),
                'Ent Type': token.ent_type_,
                'IsAlpha': token.is_alpha,
                'IsPunct': token.is_punct,
                'IsStop': token.is_stop
            }
        )
    tokes = pd.DataFrame(rows)
    tokes.fillna("", inplace=True)  # Handle NaN values
    
    # Display the DataFrame
    st.write("DataFrame Info:")
    st.write(tokes.info())
    
    if not tokes.empty:
        AgGrid(tokes)
    else:
        st.warning("The DataFrame is empty. Check your input text.")

    # Download the tokens?
    csv = convert_df(tokes)
    st.download_button(
        label="Download TOKEN DATA as CSV",
        data=csv,
        file_name='tokens.csv',
        mime='text/csv',
    )

    # Display Named Entity Information
    st.markdown('## Named Entity Information')

    # Define custom explanations for your labels
    label_explanations = {
        "CARDINAL": "A cardinal number, like 42, 3rd, etc.",
        "DATE": "A specific date or time period.",
        "EVENT": "A historical event, like a war, tournament, etc.",
        "FAC": "A building or structure.",
        "GPE": "Geopolitical entity like a country, city, or region.",
        "HABITAT": "A place where an organism lives.",
        "LANGUAGE": "A spoken or written language.",
        "LAW": "A legal document or rule.",
        "LOC": "A location or place.",
        "MONEY": "A monetary value, including units.",
        "NORP": "Nationalities or religious/political groups.",
        "ORDINAL": "An ordinal number, like 1st, 2nd, etc.",
        "ORG": "An organization, company, institution, etc.",
        "PERCENT": "A percentage.",
        "PERSON": "A person, including a fictional character.",
        "PRODUCT": "A product, usually commercial.",
        "QUANTITY": "A physical quantity, like weight, volume, etc.",
        "TAXA": "Taxonomic name for a biological organism or species.",
        "TIME": "A specific time, hour, minute, etc.",
        "WORK_OF_ART": "A creative work, such as a book, painting, etc."
    }

    entity_rows = []
    for ent in doc.ents:
        explanation = label_explanations.get(ent.label_, spacy.explain(ent.label_))  # Use custom explanation if available
        entity_rows.append({
            'Entity Text': ent.text,
            'Start Position': ent.start_char,
            'End Position': ent.end_char,
            'Label': ent.label_,
            'Explanation': explanation
        })
    entity_df = pd.DataFrame(entity_rows)
    entity_df.fillna("", inplace=True)  # Handle NaN values

    if not entity_df.empty:
        st.write(entity_df)
        # Add option to download named entity data
        entity_csv = convert_df(entity_df)
        st.download_button(
            label="Download ENTITY DATA as CSV",
            data=entity_csv,
            file_name='entities.csv',
            mime='text/csv',
        )
    else:
        st.warning("No named entities found. Check your input text.")
    
    st.text(f"Analyzed using spaCy model {DEFAULT_MODEL}")

else:
    st.markdown(":sparkles: **Paste Text Here** :sparkles: ")
    text = st.text_area("(Default text is shown)", DEFAULT_TEXT, height=200)

    nlp = spacy.load(DEFAULT_MODEL)
    ruler = nlp.add_pipe("entity_ruler", before='ner')
    ruler.from_disk(Path(__file__).parent / "../data/ansp-clean-patterns.jsonl")
    doc = nlp(text)
    
    # to plot the labels for the custom entities, we have to make a new set
    labels = list(nlp.get_pipe("ner").labels)
    for label in nlp.get_pipe("entity_ruler").labels:
        labels.append(label)
    
    colors = {"CARDINAL": "#DB9D85", "DATE": "#D0A374", "EVENT": "#C2A968", "FAC": "#B1AF64", "GPE": "#9DB469",
              "HABITAT": "#86B875", "LANGUAGE": "#6DBC86", "LAW": "#53BE98", "LOC": "#3DBEAB", "MONEY": "#39BDBC",
              "NORP": "#4CB9CC", "ORDINAL": "#69B4D8", "ORG": "#87AEDF", "PERCENT": "#A3A7E2", "PERSON": "#BB9FE0",
              "PRODUCT": "#CD99D8", "QUANTITY": "#DA95CC", "TAXA": "#E293BD", "TIME": "#E494AB", "WORK_OF_ART": "#E29898"}

    spacy_streamlit.visualize_ner(
        doc,
        labels=labels,
        colors=colors,
        title="Custom Entity Labels",
        show_table=False,
    )

    st.markdown('## Exploring the Tokens')
    # Create dataframe for data download
    rows = []
    for token in doc:
        rows.append(
            {
                'Token': token.text,
                'Lemma': token.lemma_,
                'POS': token.pos_,
                'Tag': token.tag_,
                'Dependency': token.dep_,
                'Head': token.head.text if hasattr(token.head, 'text') else str(token.head),
                'Ent Type': token.ent_type_,
                'IsAlpha': token.is_alpha,
                'IsPunct': token.is_punct,
                'IsStop': token.is_stop
            }
        )
    tokes = pd.DataFrame(rows)
    tokes.fillna("", inplace=True)  # Handle NaN values

    # Display the DataFrame
    st.write("DataFrame Info:")
    st.write(tokes.info())
    
    if not tokes.empty:
        AgGrid(tokes)
    else:
        st.warning("The DataFrame is empty. Check your input text.")
        
    # Download the tokens?
    csv = convert_df(tokes)
    st.download_button(
        label="Download TOKEN DATA as CSV",
        data=csv,
        file_name='tokens.csv',
        mime='text/csv',
    )

    # Display Named Entity Information
    st.markdown('## Named Entity Information')

    # Define custom explanations for your labels
    label_explanations = {
        "CARDINAL": "A cardinal number, like 42, 3rd, etc.",
        "DATE": "A specific date or time period.",
        "EVENT": "A historical event, like a war, tournament, etc.",
        "FAC": "A building or structure.",
        "GPE": "Geopolitical entity like a country, city, or region.",
        "HABITAT": "A place where an organism lives.",
        "LANGUAGE": "A spoken or written language.",
        "LAW": "A legal document or rule.",
        "LOC": "A location or place.",
        "MONEY": "A monetary value, including units.",
        "NORP": "Nationalities or religious/political groups.",
        "ORDINAL": "An ordinal number, like 1st, 2nd, etc.",
        "ORG": "An organization, company, institution, etc.",
        "PERCENT": "A percentage.",
        "PERSON": "A person, including a fictional character.",
        "PRODUCT": "A product, usually commercial.",
        "QUANTITY": "A physical quantity, like weight, volume, etc.",
        "TAXA": "Taxonomic name for a biological organism or species.",
        "TIME": "A specific time, hour, minute, etc.",
        "WORK_OF_ART": "A creative work, such as a book, painting, etc."
    }

    entity_rows = []
    for ent in doc.ents:
        explanation = label_explanations.get(ent.label_, spacy.explain(ent.label_))  # Use custom explanation if available
        entity_rows.append({
            'Entity Text': ent.text,
            'Start Position': ent.start_char,
            'End Position': ent.end_char,
            'Label': ent.label_,
            'Explanation': explanation
        })
    entity_df = pd.DataFrame(entity_rows)
    entity_df.fillna("", inplace=True)  # Handle NaN values

    if not entity_df.empty:
        st.write(entity_df)
        # Add option to download named entity data
        entity_csv = convert_df(entity_df)
        st.download_button(
            label="Download ENTITY DATA as CSV",
            data=entity_csv,
            file_name='entities.csv',
            mime='text/csv',
        )
    else:
        st.warning("No named entities found. Check your input text.")
        
    st.text(f"Analyzed using spaCy model {DEFAULT_MODEL}")
