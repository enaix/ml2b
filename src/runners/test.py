from src.bench import *  # We don't care about re-importing the module


class TestRunner:
    input_mode: RunnerInput = RunnerInput.DescOnly
    output_mode: RunnerOutput = RunnerOutput.CodeOnly
    runner_id: str = "test_runner"

    def __init__(self):
        pass

    # run() does not take CompetitionData, since input_mode is DescOnly
    def run(self, bench: BenchPipeline, comp: Competition, lang: Language, codelang: CodeLanguage) -> dict:
        # get description and other stuff from comp
        # call bench to execute
        # return resulting score

        code = """
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
def train_and_predict(X_train, y_train, X_test):
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled = scaler.transform(X_test[numeric_cols])
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    return model.predict_proba(X_test_scaled)[:, 1]
        """

        pass

    # if we needed to process data
    #def run(self, bench: BenchPipeline, comp: Competition, fold: CompetitionData) -> dict:
    #    pass
