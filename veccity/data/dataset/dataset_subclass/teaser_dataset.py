from .skipgram_dataset import SkipGramData
import numpy as np

class TeaserData(SkipGramData):
    def __init__(self, users, sentences, weeks, coordinates, num_ne, num_nn, indi_context, distance_threshold=5,
                 sample=1e-3):
        """
        @param sentences: all users' full trajectories, shape (num_users, seq_len)
        @param weeks: weekday indices corresponding to the sentences.
        @param coordinates: coordinates of all locations, shape (num_locations, 3), each row is (loc_index, lat, lng)
        """
        super().__init__(sentences, sample)
        self.num_ne = num_ne
        self.num_nn = num_nn
        self.indi_context = indi_context
        self.users = users
        self.weeks = weeks
        all_locations = set(coordinates[:, 0].astype(int).tolist())
        self.location_num = len(all_locations)
        lat_sin = np.sin(np.radians(coordinates[:, 1]))
        lat_cos = np.cos(np.radians(coordinates[:, 1]))
        lng_radians = np.radians(coordinates[:, 2])
        # logger = getLogger()
        # logger.info('Total {}'.format(len(coordinates)))
        # counter = 0
        # A dict mapping one location index to its non-neighbor locations.
        self.non_neighbors = {}
        for coor_row in coordinates:

            loc_index = int(coor_row[0])
            # logger.info(loc_index)
            # distance = coor_row[1:].reshape(1, 2) - coordinates[:, 1:]  # (num_loc, 2)
            # distance = np.sqrt(np.power(distance[:, 0], 2) + np.power(distance[:, 1], 2))  # (num_loc)

            lat = np.full(len(coordinates), np.radians(coor_row[1]))
            lng = np.full(len(coordinates), np.radians(coor_row[2]))
            distance = np.arccos(np.minimum(np.sin(lat) * lat_sin +
                                            np.cos(lat) * lat_cos * np.cos(lng - lng_radians), 1.)) * 6371.

            non_neighbor_indices = coordinates[:, 0][np.argwhere(distance > distance_threshold)].reshape(-1).astype(int)
            if non_neighbor_indices.shape[0] == 0:
                non_neighbor_indices = np.array([len(all_locations)], dtype=int)
            self.non_neighbors[loc_index] = non_neighbor_indices

            # counter += 1
            # if counter % 1000 == 0:
            #     logger.info('Finish {}'.format(counter))

        # logger.info('Total {}'.format(min(len(users), len(sentences))))
        # counter = 0
        # A dict mapping one user index to its all unvisited locations.
        # self.unvisited = {}
        # for user, visited in zip(users, sentences):
        #     user = int(user)
        #     user_unvisited = all_locations - set(visited)
        #     self.unvisited[user] = user_unvisited & self.unvisited.get(user, all_locations)
        #     counter += 1
        #     if counter % 1000 == 0:
        #         logger.info('Finish {}'.format(counter))

        self.visited = {}
        for user, visited in zip(users, sentences):
            user = int(user)
            self.visited[user] = set(visited) & self.visited.get(user, set())
            # counter += 1
            # if counter % 1000 == 0:
            #     logger.info('Finish {}'.format(counter))

    def gen_pos_pairs(self, window_size):
        pos_pairs = []
        # logger = getLogger()
        # logger.info('Total {}'.format(min(len(self.users), len(self.sentences), len(self.weeks))))
        # counter = 0
        for user, sentence, week in zip(self.users, self.sentences, self.weeks):
            for i in range(0, len(sentence) - (2 * window_size + 1)):
                target = sentence[i + window_size]
                target_week = 0 if week[i + window_size] in range(5) else 1
                context = sentence[i:i + window_size] + sentence[i + window_size + 1:i + 2 * window_size + 1]
                sample_ne = self.sample_unvisited(user, num_neg=self.num_ne)
                sample_nn = self.sample_non_neighbor(target, num_neg=self.num_nn)
                if self.indi_context:
                    pos_pairs += [[user, target, target_week, [c], sample_ne, sample_nn] for c in context]
                else:
                    pos_pairs.append([user, target, target_week, context, sample_ne, sample_nn])
            # counter += 1
            # if counter % 1000 == 0:
            #     logger.info('Finish {}'.format(counter))
        return pos_pairs

    def sample_unvisited(self, user, num_neg):
        # return np.random.choice(np.array(list(self.unvisited[user])), size=num_neg).tolist()
        unvisited = []
        for i in range(num_neg):
            location = np.random.randint(0, self.location_num)
            while location in self.visited[user]:
                location = np.random.randint(0, self.location_num)
            unvisited.append(location)
        return unvisited

    def sample_non_neighbor(self, target, num_neg):
        return np.random.choice(self.non_neighbors[target], size=num_neg).tolist()