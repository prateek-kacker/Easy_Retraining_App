'''
This is a gradio app for uploading excel file, annotating the data, training the model and verifying the output for the model. You can keep repeating it till the model is getting desired results.
'''
import gradio as gr
import pandas as pd

base_df = pd.DataFrame()
SENT1 = ''
SENT2 = ''
LABEL = ''
TEXT_CATEGORIES= []
MAX_TEXT_CATEGORIES=5
MAX_LENGTH = 0
DF_INDEX = 0
NEXT = 1
PREVIOUS = 0

def moving_direction(direction):
    '''
    Function to move and back in the dataframe
    '''
    global MAX_LENGTH
    global NEXT
    global PREVIOUS
    global DF_INDEX
    if DF_INDEX == MAX_LENGTH-1 and direction == 'FORWARD':
        NEXT = MAX_LENGTH-1
        PREVIOUS = DF_INDEX-1
        DF_INDEX = DF_INDEX
    elif DF_INDEX == 0 and direction == 'BACKWARD':
        PREVIOUS = 0
        NEXT = DF_INDEX+1
        DF_INDEX = DF_INDEX 
    elif direction == 'FORWARD':
        DF_INDEX = DF_INDEX+1
        NEXT = DF_INDEX+1
        PREVIOUS = DF_INDEX-1
    elif direction == 'BACKWARD':
        DF_INDEX = DF_INDEX-1
        NEXT = DF_INDEX+1
        PREVIOUS = DF_INDEX-1

    
    

     
def get_data_from_excel(excel_file):
    '''
    Function to load file when get_data_btn is executed
    '''

    global base_df
    global MAX_LENGTH
    base_df = pd.read_excel(excel_file.name)
    base_df['Annotated']=False
    base_df['Predicted']=False
    MAX_LENGTH = len(base_df)
    print(base_df)
    return base_df


def verify_entire_predicted_fn():
    pass

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
    print(TEXT_CATEGORIES)

    return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),gr.update(visible=True)]

def len_of_not_annotated_fn():
    global base_df
    number_of_testpoints = len(base_df[not base_df['Annotated']])
    return number_of_testpoints


def annotate_data_fn():
    global base_df
    global DF_INDEX
    global NEXT
    global PREVIOUS
    global TEXT_CATEGORIES

    text = base_df.iloc[DF_INDEX][SENT1]
    unannotated_text= gr.update(value = text,visible=True)
    category_radio=gr.update(choices = TEXT_CATEGORIES, visible=True)
    previous_btn = gr.update(visible=True)
    next_btn = gr.update(visible=True)
    accept_btn = gr.update(visible=True)
    return unannotated_text, category_radio, previous_btn, next_btn, accept_btn

def previous_fn():
    global base_df
    global DF_INDEX
    global PREVIOUS
    global NEXT
    moving_direction('BACKWARD')
    text = base_df.iloc[DF_INDEX][SENT1]
    category = base_df.iloc[DF_INDEX][LABEL]
    if pd.isnull(category):
        category = None
    unannotated_text= gr.update(value = text,visible=True)
    category_radio=gr.update(value = category,visible=True)
    return unannotated_text, category_radio

def next_fn():
    global base_df
    global DF_INDEX
    global PREVIOUS
    global NEXT
    moving_direction('FORWARD')
    text = base_df.iloc[DF_INDEX][SENT1]
    category = base_df.iloc[DF_INDEX][LABEL]
    if pd.isnull(category):
        category = None
    unannotated_text= gr.update(value = text,visible=True)
    category_radio=gr.update(value = category, visible=True)
    return unannotated_text, category_radio

def accept_fn(category_radio):
    global base_df
    global DF_INDEX
    global PREVIOUS
    global NEXT
    base_df.iloc[DF_INDEX, base_df.columns.get_loc(LABEL)] = category_radio
    base_df.iloc[DF_INDEX, base_df.columns.get_loc('Annotated')] = True
    moving_direction('FORWARD')
    text = base_df.iloc[DF_INDEX][SENT1]
    category = base_df.iloc[DF_INDEX][LABEL]
    if pd.isnull(category):
        category = None
    unannotated_text= gr.update(value=text,visible=True)
    category_radio=gr.update(value = category, visible=True)
    return unannotated_text, category_radio
    
def train_fn():
    pass    

def annotate_predicted_fn():
    pass

def verify_entire_fn():
    pass

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
        unannotated_text= gr.Textbox(label='Text for annotation',
                                     visible=False,interactive=True
                                     )
        category_radio=gr.Radio(TEXT_CATEGORIES,
                                label='Category',
                                info='Select the appropriate category for the text',
                                visible=False,
                                interactive=True
                                )
    with gr.Row() as row:
            previous_btn = gr.Button('Previous',
                                     visible=False,
                                     interactive=True
                                     )
            next_btn = gr.Button('Next',
                                 visible=False,
                                 interactive=True
                                 )
    accept_btn = gr.Button('Accept the current category and move next',
                           visible=False,
                           interactive=True
                           )
    annotate_unlabeled_btn.click(annotate_data_fn,
                                 inputs=[],
                                 outputs=[unannotated_text,
                                          category_radio,
                                          previous_btn,
                                          next_btn,
                                          accept_btn
                                          ]
                                )
    previous_btn.click(previous_fn, outputs=[unannotated_text,category_radio])
    next_btn.click(next_fn, outputs=[unannotated_text,category_radio])
    accept_btn.click(accept_fn, inputs = category_radio, outputs=[unannotated_text,category_radio])



    gr.Markdown(
        '''
        # Section 3 - Train the model based on past annotations
        '''
    )
    epochs_text = gr.Textbox(value='5',
                             label='Number of Epochs',
                             interactive=True
                             )
    batch_text = gr.Textbox(value='32',
                            label='Batch Size',
                            interactive=True
                            )
    train_btn = gr.Button('Train Model')
    train_end_text = gr.Textbox(value='Output data',
                                label='Training End'
                                )
    train_btn.click(train_fn,inputs=[epochs_text,batch_text],outputs=[train_end_text])



    gr.Markdown(
        """
        # Section 4 -Verify the output of the model on the data which is not annotated
        """
    )
    with gr.Row() as row:
        annotate_predicted_btn = gr.Button("Annotate Predicted Data")

    with gr.Row() as row:
        predicted_text= gr.Textbox(label='Text for confirmation on predicted data')
        train_category_radio=gr.Radio(TEXT_CATEGORIES, label='Category',info='Appropriate category for the text as predicted by the model')
    with gr.Row() as row:
            train_previous_btn = gr.Button('Previous')
            train_next_btn = gr.Button('Next')
    train_accept_btn = gr.Button('Accept the category after verification and move to next')
    annotate_predicted_btn.click(annotate_predicted_fn,
                                 inputs=[],
                                 outputs=[predicted_text,
                                          train_category_radio,
                                          train_previous_btn,
                                          train_next_btn,
                                          train_accept_btn
                                          ]
                                 )


    gr.Markdown(
        """
        # Section 5 - Verify the entire annotated data once again
        """
    )

    with gr.Row() as row:
        verify_entire_btn = gr.Button("Verify Entire Data")

    with gr.Row() as row:
        predicted_entire_text= gr.Textbox(label='Text for confirmation on entire data')
        predicted_entire_category_radio=gr.Radio(TEXT_CATEGORIES, label='Category',info='Appropriate category for the text as annotated by you')
    with gr.Row() as row:
            predict_entire_previous_btn = gr.Button('Previous')
            predict_entire_next_btn = gr.Button('Next')
    predict_entire_accept_btn = gr.Button('Accept the category after verification and move to next')
    verify_entire_btn.click(verify_entire_predicted_fn,
                            inputs=[],
                            outputs=[predicted_entire_text,
                                     predicted_entire_category_radio,
                                     predict_entire_previous_btn,
                                     predict_entire_next_btn,
                                     predict_entire_accept_btn
                                     ]
                                     )


demo.launch()
