# Subspace Uncertainty
...

## Requirements

### Packages
Create a virtual environment with `python=3.11` and install all the packages with `pip` from `requirements.txt`. The relevant environment is provided:
```
conda env create -f utils/create_env.yml
conda activate FisherSub
pip install -r utils/requirements.txt
pip install --upgrade build
python -m build ./drbayes
pip install drbayes/dist/subspace_inference-0.0.tar.gz
python -m build ./swa_gaussian
pip install swa_gaussian/dist/swag-0.0.tar.gz
```

## Running the script
[hydra](https://hydra.cc/docs/intro/) might be used later

## TODO:
1. Implement FIM for classification (softmax)
2. Implement a numerical method to obtain the dominant Hessian eigenvectors. 
3. (So far the calculated Hessian is returned as dictionary. For bigger models this Hessian should either be transformed into a vector, or the Hessian should be evaluated differently, or it is not necessary, since we calculate the eigenvectors with method 2.)



## Disclaimer
This software was developed at Physikalisch-Technische Bundesanstalt
(PTB). The software is made available "as is" free of cost. PTB assumes
no responsibility whatsoever for its use by other parties, and makes no
guarantees, expressed or implied, about its quality, reliability, safety,
suitability or any other characteristic. In no event will PTB be liable
for any direct, indirect or consequential damage arising in connection

## License
MIT License

Copyright (c) 2023 Datenanalyse und Messunsicherheit, working group 8.42, PTB Berlin

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Reference
