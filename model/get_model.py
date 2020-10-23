
def get_model(model_name):
    if model_name == 'cnn':
        from model.cnn import MODEL
    elif model_name == 'ann':
        from model.ann import MODEL
    else:
        print('import model error')
        exit()

    return MODEL