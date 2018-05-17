class Singleton(type):
  _instances = {}
  def __call__(cls, *args, **kwargs):
    if cls not in cls._instances:
      cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
    return cls._instances[cls]


class Config(object, metaclass=Singleton):
  def __init__(self):
    """init configuration
    
    dataset_path: path to dataset (raw) directory
    dir_mapping: directory mapping for naming convenience
    """
    
    self.config = dict()
    
    self.dataset_path = {
      'market1501': '/home/penggao/data/reid/market1501',
      'duke': '/home/penggao/data/reid/duke',
      'cuhk03': '/home/penggao/data/reid/cuhk03'
    }

    self.dir_mapping = {
      'bounding_box_train': 'train_all',
      'bounding_box_test': 'gallery',
      'query': 'query'
    }

  def __call__(self, member):
    """Make class callable"""
    assert member in self.__dict__.keys(), "config info for %s doesn't exsit" % member
    return self.__dict__[member]  

  def __setattr__(self, name, value):
    """Edit attributes"""
    object.__setattr__(self, name, value)