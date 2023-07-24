class Nodes:
    nodes_list = []
    nodes_painted = []
    colour_base = [0]
    colours_used= []
    nodes_df = []
    nodes_n_connections_nd = []

    def __init__(self, node_id, available_colours=None, colour=None):
        if available_colours is None:
            available_colours = [0]
        Nodes.nodes_list.append(self)
        self.node_id = node_id
        self.connected_nodes = []
        self.colour = colour
        self.available_colours = available_colours
        self.decision_restrictions = []
        self.added_colour = False
        Nodes.nodes_painted.append(False)

    def add_connection(self, id_other_node):
        self.connected_nodes.append(id_other_node)

    def add_available_colour(self, colour):
        self.available_colours.append(colour)

    def set_colour(self, colour):
        self.colour = colour
        Nodes.nodes_painted[self.node_id] = True
        for i in self.connected_nodes:  # Propago
            if colour in Nodes.nodes_list[i].available_colours:
                Nodes.nodes_list[i].available_colours.remove(colour)
        if colour not in Nodes.colours_used:
            Nodes.colours_used.append(colour)
            self.added_colour = True

    def possible_colours(self):  # List of colours available, not decision restricted and allowing only 1 of not used
        # print('base: ', Nodes.colour_base)
        # print('restrictions: ', self.decision_restrictions)
        # print('available: ', self.available_colours)
        # print('used: ', Nodes.colours_used)
        return [i for i in self.available_colours if (i not in self.decision_restrictions) & (i not in [j for j in Nodes.colour_base if j not in Nodes.colours_used][1:])]

    def set_possible_colour(self):  # Sets the colour to the first possible colour not restricted by decisions
        possible_colours = self.possible_colours()
        number_available = len(possible_colours)
        if number_available > 0:
            colour = possible_colours[0]
            self.set_colour(colour)
            Nodes.nodes_painted[self.node_id] = True  # The node is marked as colored
            for i in self.connected_nodes:  # Propago
                if colour in Nodes.nodes_list[i].available_colours:
                    Nodes.nodes_list[i].available_colours.remove(colour)
            return True, number_available

        else:
            return False, number_available  # With this false we need to go back to the previous decision and make it a restriction.

    def reset_colour(self):
        self.colour = None
        self.available_colours = Nodes.colour_base.copy()
        self.added_colour = False
        Nodes.nodes_painted[self.node_id] = False

    def restore_colour(self):
        colour = self.colour
        self.colour = None
        Nodes.nodes_painted[self.node_id] = False
        if self.added_colour:
            Nodes.colours_used.remove(colour)
            for i in self.connected_nodes:
                Nodes.nodes_list[i].available_colours.append(colour)
            self.added_colour = False
        else:
            for i in self.connected_nodes:
                for j in Nodes.nodes_list[i].connected_nodes:
                    if Nodes.nodes_list[j].colour == colour:
                        break
                else:
                    Nodes.nodes_list[i].available_colours.append(colour)

    def set_available_as_base(self):
        self.available_colours = Nodes.colour_base.copy()
        #print(self.node_id, self.available_colours, Nodes.colour_base)

    def add_decision_restriction(self, colour):
        self.decision_restrictions.append(colour)

    def reset_decision_restrictions(self):
        #print(f'node {self.node_id} restrictions reseted')
        self.decision_restrictions = []

    @classmethod
    def calc_df_nd(cls):
        cls.nodes_df = [len(i.available_colours) for i in cls.nodes_list]
        #cls.nodes_df = [len(i.possible_colours()) for i in cls.nodes_list]
        cls.nodes_n_connections_nd = [sum([Nodes.nodes_list[j].colour == None for j in i.connected_nodes]) for i in cls.nodes_list]
