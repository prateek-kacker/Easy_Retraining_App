import gradio as gr
import pandas as pd

base_df = pd.DataFrame()


def get_data_from_excel(excel_file):
    '''
    Function to load file when get_data_btn is executed
    '''
    
    global base_df
    base_df = pd.read_excel(excel_file.name)
    print(base_df)
    return base_df


def show_headers():
    string = ''
    column_string = list(base_df.columns)
    return column_string.join(', ')

with gr.Blocks() as demo:
    with gr.Row() as row:
        get_data_btn = gr.UploadButton("Get Data from excel")
        show_df_text = gr.DataFrame()
        get_data_btn.upload(fn=get_data_from_excel,inputs=get_data_btn,outputs=show_df_text)
    with gr.Row() as row:
        annotate_unlabeled_btn = gr.Button("Annotate Unlabeled Data")
        annotate_predicted_data_btn = gr.Button("Annotate predicted Data")
        train_btn = gr.Button('Train Model')


demo.launch()
