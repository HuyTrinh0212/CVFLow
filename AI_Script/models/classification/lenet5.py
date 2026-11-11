from AI_Script.models.base_model import BaseModel
from AI_Script.models.registry_model import ModelRegistry

@ModelRegistry.register("lenet5")
class LENET5(BaseModel):
    def __init__(self, config=None, debug_mode=False):
        super().__init__(config, debug_mode=debug_mode)

    # Model predict
    def predict(self, input_data):
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_data})
        if self.debug_mode:
            profile = self.session.end_profiling()
            return (outputs, profile)
        return outputs

