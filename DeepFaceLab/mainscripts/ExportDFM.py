import models


def main(model_class_name, saved_models_path):
    model = models.import_model(model_class_name)(
                        is_exporting=True,
                        saved_models_path=saved_models_path,
                        cpu_only=True)
    model.export_dfm () 