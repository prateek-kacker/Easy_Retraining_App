import gradio as gr
import pandas as pd

base_df = pd.DataFrame()
SENT1 = ''
SENT2 = ''
LABEL = ''
TEXT_CATEGORIES= []

def get_data_from_excel(excel_file):
    '''
    Function to load file when get_data_btn is executed
    '''

    global base_df
    base_df = pd.read_excel(excel_file.name)
    print(base_df)
    return base_df


def show_headers():
    '''
    This function shows headers in excel
    '''

    column_string = list(base_df.columns)
    return ', '.join(column_string)


def agree_fn(sent1_header, sent2_header, label_header,text_categories):
    '''
    Function for agree btn 
    '''

    global SENT1
    global SENT2
    global LABEL
    global TEXT_CATEGORIES
    SENT1 = sent1_header
    if sent2_header.strip() == '':
        SENT2 = sent2_header
    LABEL = label_header
    TEXT_CATEGORIES = text_categories.split(',')
    TEXT_CATEGORIES = [category.strip() for category in TEXT_CATEGORIES]

    return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),gr.update(visible=True)]


with gr.Blocks() as demo:
    with gr.Row() as row:
        get_data_btn = gr.UploadButton("Get Data from excel")
        show_df_text = gr.DataFrame()
        get_data_btn.upload(fn=get_data_from_excel,
                            inputs=get_data_btn, outputs=show_df_text)
    with gr.Row() as row:
        show_headers_btn = gr.Button("Show headers ")
        show_header_text = gr.Textbox()
        show_headers_btn.click(fn=show_headers, outputs=show_header_text)
    with gr.Row() as row:
        header_sent1 = gr.Textbox(
            label='Header for Sentence1', interactive=True)
        header_sent2 = gr.Textbox(
            label='Header for Sentence2 or leave blank', interactive=True)
        header_label = gr.Textbox(label="Header for Label", interactive=True)
        categories_text = gr.Textbox(label='Categories seperated by comma',interactive=True)

    agree_btn = gr.Button('Fill the Header and accept it')
    agree_accepted_btn = gr.Button("Agreed and Accepted", visible=False)
    agree_btn.click(agree_fn, inputs=[header_sent1, header_sent2, header_label,categories_text], outputs=[
                    header_sent1, header_sent2, header_label, agree_btn,agree_accepted_btn])

    with gr.Row() as row:
        annotate_unlabeled_btn = gr.Button("Annotate Unlabeled Data")


    with gr.Row() as row:
        annotate_predicted_data_btn = gr.Button("Annotate predicted Data")
        train_btn = gr.Button('Train Model')


demo.launch()
