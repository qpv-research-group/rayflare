from solcore.structure import Layer

class Structure(list):
    # both interfaces and bulk layers

    def __init__(self, *args, **kwargs):
        super(Structure, self).__init__(*args)
        self.__dict__.update(kwargs)
        self.labels = [None] * len(self)

    # def append(self, new_layer, layer_label=None, repeats=1):
    #     # Pass the arguments to the superclass for extending
    #     for i in range(repeats):
    #         # Extend the structure labels
    #         self.labels.append(layer_label)
    #         super(Structure, self).append(new_layer)
    #
    # def append_multiple(self, layers, layer_labels=None, repeats=1):
    #
    #     assert type(layers) == type([]), "`append_multiple` only accepts lists for the first argument."
    #
    #     if layer_labels is not None:
    #         assert len(layers) == len(
    #             layer_labels), "When using `layer_labels` keyword a label must be specified for each layer added i.e. layers and layer_labels must have the same number of elements.  Either fix this or simply do not assign any labels (i.e. layer_labels=None)."
    #
    #     for i in range(repeats):
    #         # Extend the structure by appending layers
    #         self.extend(layers)
    #
    #         # Extend the structure labels by appending an equal number of None values
    #         # or by appending the actual labels.
    #         if layer_labels is None:
    #             self.labels.extend([None] * len(layers))
    #         else:
    #             self.labels.extend(layer_labels)

    def __str__(self):

        layer_info = ["  {} {}".format(
            layer,
            self.labels[i] if self.labels[i] is not None else "",
        ) for i, (layer, label) in enumerate(zip(self, self.labels))]

        return "<Structure object\n{}\n{}>".format(str(self.__dict__), "\n".join(layer_info))



class BulkLayer:
    """ Class that stores the information about layers of materials, such as thickness and composition.
    It is the building block of the 'Structures' """

    def __init__(self, width, material, **kwargs):
        """ Layer class constructor.

        """
        self.width = width
        self.material = material
        self.__dict__.update(kwargs)


class Interface:

    def __init__(self, method, layers=None, texture=None, prof_layers=None, coherent=True,  **kwargs):
        """ Layer class constructor.

        """
        self.method = method
        self.__dict__.update(kwargs)
        self.layers = layers
        self.texture = texture # for ray tracing
        self.prof_layers = prof_layers # in which layers of the interface (1-indexed) should absorption be calculated?
        self.materials = []
        self.n_depths = []
        self.widths = []
        self.coherent = coherent

        if layers is not None:
            for element in layers:
                if isinstance(element, Layer):
                    self.materials.append(element.material)
                    self.widths.append(element.width)

                else:
                    self.widths.append(element[0]*1e-9)
                    self.materials.append(element[1:3])


class Texture:

    def __init__(self, texture):
        self.texture = texture


class RTgroup:

    def __init__(self, textures, materials=None, widths=None):
        self.materials = materials
        self.textures = textures
        self.widths = widths

