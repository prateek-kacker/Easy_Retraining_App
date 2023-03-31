import gradio as gr
import pandas as pd

base_df = pd.DataFrame()
SENT1 = ''
SENT2 = ''
LABEL = ''
TEXT_CATEGORIES= []
MAX_TEXT_CATEGORIES=5

def get_data_from_excel(excel_file):
    '''
    Function to load file when get_data_btn is executed
    '''

    global base_df
    base_df = pd.read_excel(excel_file.name)
    base_df['Annoated']=False
    base_df['Predicted']=False

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

def len_of_not_annotated_fn():
    global base_df
    number_of_testpoints = len(base_df[not base_df['Annotated']])
    return number_of_testpoints


def annotate_data_fn():
    global base_df
    
    

with gr.Blocks() as demo:
    gr.Markdown(
    """
    # Section 1 - Data input and Headers Definition
    """)
        
    with gr.Row() as row:
        get_data_btn = gr.UploadButton("Get Data from excel")
        show_df_text = gr.DataFrame()
        get_data_btn.upload(fn=get_data_from_excel,
                            inputs=get_data_btn, 
                            outputs=show_df_text
                            )
    
    with gr.Row() as row:
        show_headers_btn = gr.Button("Show headers ")
        show_header_text = gr.Textbox()
        show_headers_btn.click(fn=show_headers, 
                               outputs=show_header_text)
        
    with gr.Row() as row:
        header_sent1 = gr.Textbox(label='Header for Sentence1', 
                                  interactive=True
                                  )
        header_sent2 = gr.Textbox(label='Header for Sentence2 or leave blank', 
                                  interactive=True
                                  )
        header_label = gr.Textbox(label="Header for Label",
                                  interactive=True
                                  )
        categories_text = gr.Textbox(label='Categories seperated by comma',
                                     interactive=True
                                     )

    agree_btn = gr.Button('Fill the Header and accept it')
    agree_accepted_btn = gr.Button("Agreed and Accepted",
                                   visible=False
                                   )
    agree_btn.click(agree_fn, 
                    inputs=[header_sent1,
                            header_sent2,
                            header_label,
                            categories_text
                            ],
                    outputs=[header_sent1,
                             header_sent2, 
                             header_label, 
                             agree_btn,
                             agree_accepted_btn
                             ]
                    )

    gr.Markdown(
    """
    # Section 2 - Annotate datapoints which you want to provide for training
    """)

    with gr.Row() as row:
        annotate_unlabeled_btn = gr.Button("Annotate Unlabeled Data")

    with gr.Row() as row:
        unannotated_text= gr.Textbox(label='Text for annotation')
        category_radio=gr.Radio(TEXT_CATEGORIES, label='Category',info='Select the appropriate category for the text')
    with gr.Row() as row:
            previous_btn = gr.Button('Previous')
            next_btn = gr.Button('Next')
    accept_btn = gr.Button('Accept the current category and move next')
    annotate_unlabeled_btn.click(annotate_data_fn,inputs=[],outputs=[unannotated_text,category_radio,accept_btn])

    gr.Markdown(
        '''
        # Section 3 - Train the model based on past annotations
        '''
    )






    gr.Markdown(
        """
        # Section 4 -Verify the output of the model on the data which is not annotated
        """
    )


    gr.Markdown(
        """
        # Section 5 - Verify the entire annotated data once again
        """
    )
    with gr.Row() as row:
        annotate_predicted_data_btn = gr.Button("Annotate predicted Data")
        train_btn = gr.Button('Train Model')


demo.launch()
