from codegen_sources.preprocessing.dataset_modes.monolingual_mode import MonolingualMode
from codegen_sources.preprocessing.dataset_modes.obfuscation_mode import ObfuscationMode
from iren.dataset_builder.source_dataset_mode import SourceDatasetMode


class SourceObfuscationMode(SourceDatasetMode, ObfuscationMode):
    pass


class SourceMonolingualMode(SourceDatasetMode, MonolingualMode):
    pass
