STAGES = [
    {"name": "Stage 1: Beginner", "gravity": 0.6, "friction": 2.50, "fall_penalty": 0.1, "promote_at": 100},
    {"name": "Stage 2: Intermediate", "gravity": 0.75, "friction": 2.00, "fall_penalty": 0.5, "promote_at": 150},
    {"name": "Stage 3: Advanced", "gravity": 0.9, "friction": 1.50, "fall_penalty": 0.5, "promote_at": 200},
    {"name": "Stage 4: Expert", "gravity": 1.0, "friction": 1.00, "fall_penalty": 0.75, "promote_at": None},
]

WINDOW = 100

class Curriculum:
    def __init__(self, stages):
        self.stages = stages
        self.index = 0
        self.recent_scores = []

    @property
    def stage(self):
        return self.stages[self.index]

    def env_settings(self):
        s = self.stage
        return {"gravity": s["gravity"], "friction": s["friction"], "fall_penalty": s["fall_penalty"]}

    def add_scores(self, scores):
        self.recent_scores = (self.recent_scores + scores)[-WINDOW:]

    def average_score(self):
        if not self.recent_scores:
            return 0
        return sum(self.recent_scores) / len(self.recent_scores)

    def ready_to_promote(self):
        if self.index >= len(self.stages) - 1 or self.stage["promote_at"] is None:
            return False
        if len(self.recent_scores) < WINDOW:
            return False
        return self.average_score() >= self.stage["promote_at"]

    def promote(self):
        self.index += 1
        self.recent_scores = []