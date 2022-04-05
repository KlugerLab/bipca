import unittest
from bipca.base import *
from utils import raises
from sklearn.exceptions import NotFittedError

class FooFitted():
    def remove_fit(self):
        if hasattr(self,'fit_'):
            del self.fit_
    def set_fit(self):
        if not hasattr(self, 'fit_'):
            self.fit_ = True
        else:
            self.fit_ = not self.fit_

    @fitted
    def fitted_fn(self):
        return True

    @fitted_property
    def fitted_prop(self):
        return self._fitted_prop
    @fitted_prop.setter
    def fitted_prop(self,val):
        self._fitted_prop = val

class TestFitted(unittest.TestCase):
    
    obj = FooFitted()

    @raises(AttributeError,startswith='The requested function')
    def test_object_does_not_have_fit(self):
        self.obj.remove_fit()
        self.obj.fitted_fn()

    def test_object_is_fitted(self):
        self.obj.remove_fit()
        self.obj.set_fit()
        return self.obj.fitted_fn()

    @raises(NotFittedError)
    def test_object_is_not_fitted(self):
        self.obj.remove_fit()
        self.obj.set_fit()
        self.obj.set_fit()
        return self.obj.fitted_fn()

    @raises(NotFittedError)
    def test_prop_set_not_fitted(self):
        self.obj.fitted_prop = True
        return self.obj.fitted_prop

    def test_prop_set_fitted(self):
        self.obj.set_fit()
        self.obj.fitted_prop = True
        return self.obj.fitted_prop

    @raises(AttributeError)
    def test_prop_set_fitted_noattr(self):
        self.obj.remove_fit()
        self.obj.set_fit()
        self.obj.set_fit()
        if hasattr(self.obj, '_fitted_prop'):
            del self.obj._fitted_prop 
        return self.obj.fitted_prop