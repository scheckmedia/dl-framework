
import inspect
import os
import shutil
import sys
from collections import OrderedDict
from enum import Enum
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

import tensorflow as tf
import yaml
from keras_autodoc import DocumentationGenerator

from dlf.core.registry import (FRAMEWORK_CALLBACKS, FRAMEWORK_DATA_GENERATORS,
                               FRAMEWORK_EVALUATORS, FRAMEWORK_LOSSES,
                               FRAMEWORK_METRICS, FRAMEWORK_MODELS,
                               FRAMEWORK_PREPROCESSING_METHODS)

sys.path.append('..')


class CustomDocumentationGenerator(DocumentationGenerator):
    _obj = None

    def process_docstring(self, docstring):
        if self._obj is not None and 'core' not in inspect.getmodule(
                self._obj).__name__:
            argspec = inspect.getargspec(self._obj)
        return super().process_docstring(docstring)

    def process_signature(self, signature):
        return super().process_signature(signature)

    def _render_from_object(self, object_, signature_override):
        self._obj = object_
        return super()._render_from_object(object_, signature_override)


def build_core_modules():
    specs = find_spec('dlf.core.builder')
    path = Path(specs.origin).parent
    module_list = {}
    for module_file in path.glob('*.py'):
        if module_file.name == '__init__.py':
            continue

        module_file = str(module_file)
        module_file = module_file[module_file.index('/dlf') + 1:-3]
        md_path = "{}.md".format(module_file)
        if not md_path in module_list:
            module_list[md_path] = []

        module_file = module_file.replace('../', '')
        module_file = module_file.replace(os.path.sep, '.')
        module = import_module(module_file)
        members = inspect.getmembers(module)
        for name, member in members:
            if inspect.isbuiltin(member):
                continue

            if not hasattr(member, '__module__'):
                continue

            if module_file not in member.__module__:
                continue

            module_list[md_path] += ["{}.{}".format(module_file, name)]

            if inspect.isclass(member):
                for cls_method_name, cls_method in inspect.getmembers(
                        member, inspect.isroutine):
                    if cls_method_name != '__call__' and cls_method_name.startswith(
                            '_'):
                        continue

                    if not cls_method.__doc__:
                        continue

                    module_list[md_path] += [
                        "{}.{}.{}".format(module_file, name, cls_method_name)]

    return module_list


def build_modules():
    registry_types = list(
        filter(lambda x: x.startswith('FRAMEWORK'), globals().keys()))

    pages = {}
    for reg in registry_types:
        cache = {}
        path = Path(reg.lower().replace('framework_', 'dlf/'))
        for name, obj in globals()[reg].items():
            if obj.__name__ in cache:
                continue

            cache[obj.__name__] = ""
            item_path = str(path / (obj.__name__ + '.md'))
            if item_path not in pages:
                pages[item_path] = []
            mod = "{}.{}".format(obj.__module__, obj.__name__)

            # if inspect.isclass(obj):
            #     mod += '.__init__'

            pages[item_path] += [mod]

    return pages


def build_nested_navigation(element, md_path, parent):
    level = len(element)
    title = element[0].replace('_', ' ').capitalize()
    if level != 1:
        found = False
        child = []

        for p in parent:
            if title in p.keys():
                child = p[title]
                found = True
                break

        if not found:
            parent += [{title: child}]

        build_nested_navigation(element[1:], md_path, child)
    else:
        parent += [{element[0]: md_path}]


core_pages = build_core_modules()
module_pages = build_modules()

pages = {**core_pages, **module_pages}
path = Path(__file__).parent


for t in ['losses', 'metrics']:
    md = 'dlf/{}/Keras.md'.format(t)
    pages[md] = []
    mod = import_module('tensorflow.keras.{}'.format(t))
    for keras_metric in dir(mod):
        if keras_metric.startswith('_'):
            continue

        pages[md] += [
            'tensorflow.keras.{}.{}'.format(t, keras_metric)]


with open(path / 'autogen.yml') as stream:
    config = yaml.load(stream, Loader=yaml.BaseLoader)
    nav = []
    for k, _ in pages.items():
        splitted = k.replace('.md', '').split('/')
        build_nested_navigation(splitted[1:], k, nav)

    index = [idx for idx, x in enumerate(config['nav']) if list(x.keys())[0]
             == config["autogen_after"]][0]

    final_list = config['nav'][0:index + 1]
    final_list.extend(nav)
    final_list.extend(config['nav'][index + 1:])
    config['nav'] = final_list

    mdconfig = config.copy()
    mdconfig.pop('autogen_after', None)

with open(path / 'mkdocs.yml', 'w') as out:
    yaml.dump(config, out)

# print(find_spec('dlf.core.builder'))
# builders = ['dlf.core.builder.' +
#            x for x in list(dir(builder)) if x.startswith('build')]


# pages = {'dlf/core/builder.md': [
#     'dlf.core.model.ModelWrapper', 'dlf.core.model.ModelWrapper.__init__']}

# pages = {'dlf/core/core.md': ['keras.layers.Dense', 'keras.layers.Flatten'],
#          'callbacks.md': ['keras.callbacks.TensorBoard']}

shutil.copyfile(path / '..' / 'README.md', path / 'tmpl' / 'index.md')

doc_generator = CustomDocumentationGenerator(
    pages, template_dir=str(path / 'tmpl'))
doc_generator.generate(str(path / 'sources'))

shutil.rmtree(path / '..' / 'docs')
shutil.copytree(path / 'site', path / '..' / 'docs')

# if (path / 'sources' / 'framework').exists():
#     shutil.rmtree(path / 'sources' / 'framework')

# shutil.copytree(path / 'tmp' / 'framework',
#                 path / 'sources' / 'framework')
# shutil.rmtree(path / 'tmp')
