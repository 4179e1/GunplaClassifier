#!/bin/bash

tar zcvf gunpla_classifier.tgz app.py model.py dataset.py requirements.txt models/pretrained_vit2.pt models/efficentnet_b2_argument.pt data/playground data/gunpla.tgz
