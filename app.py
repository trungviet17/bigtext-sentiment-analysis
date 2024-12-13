import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
from model.deepmodel.infer import SentimentPredictorONNX
import pickle


curr_dir = os.path.dirname(os.path.abspath(__file__))

def plot_accuracy(csv_path, step_column, accuracy_columns, title, xlabel, ylabel):
    """
    Helper function to plot accuracy from a CSV file.
    
    Parameters:
        csv_path (str): Path to the CSV file.
        step_column (str): Column for x-axis values.
        accuracy_columns (list): List of columns to plot for y-axis.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    data = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in accuracy_columns:
        ax.plot(data[step_column], data[col], marker='o', label=col.split(" - ")[0])

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(title="Model", fontsize=12)
    ax.grid(False)

    spines_to_remove = ['top', 'right']
    for spine in spines_to_remove:
        ax.spines[spine].set_visible(False)
    st.pyplot(fig)

def result():
    st.title('Text Sentiment Analysis Result')

    models = {
        "Naive Bayes": os.path.join(curr_dir, 'dashboard', 'naive_bayes.png'),
        "Naive Bayes with stop word": os.path.join(curr_dir, 'dashboard', 'nb_stopword.png'),
        "Naive Bayes with TFIDF": os.path.join(curr_dir, 'dashboard', 'nb_tfidf.png'),
        "LSTM": os.path.join(curr_dir, 'dashboard', 'lstm'),
        'LR': os.path.join(curr_dir,'dashboard','output.png')
    }

    selected = st.selectbox("Select a Model", list(models.keys()))
    if selected != 'LSTM':
        image_path = models[selected]
        image = Image.open(image_path)
        st.image(image, caption=f"{selected} Result", use_container_width=True)
    else:
        lstm_dir = os.path.join(curr_dir, 'dashboard', 'lstm')
        train_acc_csv = os.path.join(lstm_dir, 'train_acc.csv')
        train_loss_csv = os.path.join(lstm_dir, 'train_loss.csv')
        train_f1_csv = os.path.join(lstm_dir, 'train_f1.csv')
        val_acc_csv = os.path.join(lstm_dir, 'val_acc.csv')
        val_loss_csv = os.path.join(lstm_dir, "val_loss.csv")
        val_f1_csv = os.path.join(lstm_dir, "val_f1.csv")


        step_column = "trainer/global_step"
        train_acc_columns = [
            "bilstm_logging - train_accuracy_epoch",
            "salstm_logging - train_accuracy_epoch",
            "lstm_logging - train_accuracy_epoch",
        ]
        train_loss_columns = [
            "bilstm_logging - train_loss_epoch",
            "salstm_logging - train_loss_epoch",
            "lstm_logging - train_loss_epoch",
        ]
        train_f1_columns = [
            "bilstm_logging - train_f1_score_epoch",
            "salstm_logging - train_f1_score_epoch",
            "lstm_logging - train_f1_score_epoch",
        ]
        val_acc_columns = [
            "bilstm_logging - val_accuracy",
            "salstm_logging - val_accuracy",
            "lstm_logging - val_accuracy",
        ]
        val_loss_columns = [
            "bilstm_logging - val_loss",
            "salstm_logging - val_loss",
            "lstm_logging - val_loss",
        ]
        val_f1_columns = [
            "bilstm_logging - val_f1_score",
            "salstm_logging - val_f1_score",
            "lstm_logging - val_f1_score",
        ]


        # Plot training accuracy
        plot_accuracy(
            csv_path=train_acc_csv,
            step_column=step_column,
            accuracy_columns=train_acc_columns,
            title="Training Accuracy by Step",
            xlabel="Global Step",
            ylabel="Training Accuracy"
        )

        # Plot training loss
        plot_accuracy(
            csv_path=train_loss_csv,
            step_column=step_column,
            accuracy_columns=train_loss_columns,
            title="Training Loss by Step",
            xlabel="Global Step",
            ylabel="Training Loss"
        )

        # Plot training f1
        plot_accuracy(
            csv_path=train_f1_csv,
            step_column=step_column,
            accuracy_columns=train_f1_columns,
            title="Training F1-Score by Step",
            xlabel="Global Step",
            ylabel="Training F1-Score"
        )

        # Plot validation accuracy
        plot_accuracy(
            csv_path=val_acc_csv,
            step_column=step_column,
            accuracy_columns=val_acc_columns,
            title="Validation Accuracy by Step",
            xlabel="Global Step",
            ylabel="Validation Accuracy"
        )

        # Plot validation loss
        plot_accuracy(
            csv_path=val_loss_csv,
            step_column="Step",
            accuracy_columns=val_loss_columns,
            title="Validation Loss by Step",
            xlabel="Global Step",
            ylabel="Validation Loss"
        )

        # Plot validation f1
        plot_accuracy(
            csv_path=val_f1_csv,
            step_column=step_column,
            accuracy_columns=val_f1_columns,
            title="Validation F1-Score by Step",
            xlabel="Global Step",
            ylabel="Validation F1-Score"
        )

def inference():
    st.title("Inference")
    model_dir = os.path.join(curr_dir, "model", "deepmodel", "model_cpt")

    label_encoder_dir = os.path.join(model_dir, "label_encoder.pkl")
    with open(label_encoder_dir, "rb") as f:
        label_encoder = pickle.load(f)
    
    tokenizer_dir = os.path.join(model_dir, "tokenizer.pkl")
    with open(tokenizer_dir, "rb") as f:
        tokenizer = pickle.load(f)


    lstms = {
        "BiLSTM": os.path.join(model_dir, "bilstm_classifier.onnx"),
        "LSTM": os.path.join(model_dir, "lstm_classifier.onnx"),
        "SALSTM": os.path.join(model_dir, "salstm_classifier.onnx"),
    }

    
    selected = st.selectbox("Select a Model", list(lstms.keys()), key="inference_model_selector")

    
    if "previous_inference_model" not in st.session_state or st.session_state.previous_inference_model != selected:
        st.session_state.previous_inference_model = selected
        st.session_state.prediction = None  

    model_path = lstms[selected]
    predictor = SentimentPredictorONNX(
        model_path=model_path,
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        max_len=50,
    )

    text_input = st.text_input("Enter a sentence:", key="text_input")
    
    if st.button("Predict"):
        if text_input.strip():
            pred = predictor.predict(str(text_input))
            st.session_state.prediction = pred  # Save prediction
        else:
            st.session_state.prediction = "Please enter valid text."

    # Display prediction if available
    if st.session_state.prediction:
        st.write("Predicted Sentiment:", st.session_state.prediction)
    
pg = st.navigation([st.Page(result, title='Result'), st.Page(inference, title='Inference')])   
pg.run()