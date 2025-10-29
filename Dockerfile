FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN pip3 install monai==1.4.0
RUN pip3 install TotalSegmentator
RUN pip3 install nnunetv2==2.4.2
RUN pip3 install blosc2
RUN pip3 install acvl-utils==0.2