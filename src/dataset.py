class Documents:
    """
    In this class, I implement a document object, which store attributes of a documents such as title, author, detail and text.
    ...
    Attributes:
    ----------
    title: str
        title of a document
    author: str
        author of a document
    detail: str
        detail of a document(.B in file)
    text: str | list
        text of a document which represent as a string but list after tokenizing
    """
    def __init__(self, index, title, author, detail, text):
        self.index: str = index
        self.title: str = title
        self.author: str = author
        self.detail: str = detail
        self.text: str | list = text

    def __str__(self):
        return f"title: {self.title}\nauthor: {self.author}\ndetail: {self.detail}text: \n{self.text}\n"

    def __repr__(self):
        return f"title: {self.title}\nauthor: {self.author}\ndetail: {self.detail}text: \n{self.text}\n"