from dvc.repo import Repo

def track_with_dvc(file_path):
    repo = Repo()
    repo.add(file_path)
    repo.push()