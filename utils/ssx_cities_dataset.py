class SSXCitiesDataset(InMemoryDataset):
    ssx_url = 'https://github.com/spacesyntax/OpenMapping/releases/download/gb-v1/ssx_OpenMapping_gb_v1_gpkg.zip'
    ssx_unzip_path = '/OpenMapping-gb-v1_gpkg/gpkg/ssx_openmapping_gb_v1.gpkg'
    gva_url = 'https://www.ons.gov.uk/file?uri=%2feconomy%2fgrossvalueaddedgva%2fdatasets%2fregionalgvaibylocalauthorityintheuk%2f1997to2015/regionalgvaibylainuk.xls'
    
    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.name = name
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['regionalgvaibylainuk.xls', 'ssx_openmapping_gb_v1.gpkg']

    @property
    def processed_file_names(self):
        return ['data.pt']
    
    @property
    def num_node_labels(self) -> int:
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self) -> int:
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self) -> int:
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self) -> int:
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    def download(self):
        # Download to `self.raw_dir`.
        folder = osp.join(self.root, self.name)
        path = download_url(ssx_url, folder)
        extract_zip(path, self.raw_dir)
        path_to_curr = osp.join(self.raw_dir, )
        os.rename("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(folder, self.name), self.raw_dir)
        
        download_url(ssx_url, )
        download_url(gva_url, self.raw_dir)

    def process(self):
        # Read data into huge `Data` list.
        gdf = gpd.read_file(self.raw_file_names[0], ignore_fields=ignore_fields)
        gva_df = pd.read_excel(self.raw_file_names[1]
        data_list = load_graphs(gdf, gva_df)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])