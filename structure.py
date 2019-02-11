from ray_tracing.rt import RTSurface

class Structure(list):

    def __init__(self, *args, **kwargs):
        super(Structure, self).__init__(*args)
        self.__dict__.update(kwargs)
        self.labels = [None] * len(self)

    def append(self, new_layer, layer_label=None, repeats=1):
        # Pass the arguments to the superclass for extending
        for i in range(repeats):
            # Extend the structure labels
            self.labels.append(layer_label)
            super(Structure, self).append(new_layer)

    def append_multiple(self, layers, layer_labels=None, repeats=1):

        assert type(layers) == type([]), "`append_multiple` only accepts lists for the first argument."

        if layer_labels is not None:
            assert len(layers) == len(
                layer_labels), "When using `layer_labels` keyword a label must be specified for each layer added i.e. layers and layer_labels must have the same number of elements.  Either fix this or simply do not assign any labels (i.e. layer_labels=None)."

        for i in range(repeats):
            # Extend the structure by appending layers
            self.extend(layers)

            # Extend the structure labels by appending an equal number of None values
            # or by appending the actual labels.
            if layer_labels is None:
                self.labels.extend([None] * len(layers))
            else:
                self.labels.extend(layer_labels)

    def __str__(self):

        layer_info = ["  {} {}".format(
            layer,
            self.labels[i] if self.labels[i] is not None else "",
        ) for i, (layer, label) in enumerate(zip(self, self.labels))]

        return "<Structure object\n{}\n{}>".format(str(self.__dict__), "\n".join(layer_info))

    def width(self):
        return sum([layer.width for layer in self])


class Layer:
    """ Class that stores the information about layers of materials, such as thickness and composition.
    It is the building block of the 'Structures' """

    def __init__(self, width, material, **kwargs):
        """ Layer class constructor.

        """
        self.width = width
        self.material = material
        self.__dict__.update(kwargs)


class Group:

    def __init__(self, layers, method, **kwargs):
        """ Layer class constructor.

        """
        self.method = method
        self.__dict__.update(kwargs)
        self.layers = layers
        self.interfaces = []
        self.materials = []
        self.n_depths = []
        self.widths = []

        cum_width = 0

        for i, element in enumerate(layers):
            if type(element) == Interface:
                Points = element.texture.Points
                Points[:,2] = Points[:,2] - cum_width
                self.interfaces.append(RTSurface(Points))

            if type(element) == Layer:
                cum_width = cum_width + element.width*1e6
                self.materials.append(element.material)
                #self.n_depths.append(element.n_depths)
                self.widths.append(element.width)


class Interface:

    def __init__(self, texture):
        self.texture = texture





