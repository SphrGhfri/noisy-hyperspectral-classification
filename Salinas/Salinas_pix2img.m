clc
clear all

load Salinas_corrected.mat
load Salinas_gt.mat

noise_level=[0.005 0.010 0.015 0.020 0.025];
noise=["noiseless","gaussian","saltPepper","stripe"] ;
noise_name=["noise5","noise10","noise15","noise20","noise25"] ;
mkdir('Salinas_pix2img');
cd('Salinas_pix2img')

for k=1:4
    mkdir(convertStringsToChars(noise(k)))
    cd(convertStringsToChars(noise(k)))
for z=1:5
    Data = salinas_corrected;
    label = salinas_gt;
    %noiseless
        if k==1
          Data2=[];
        for i=1:204
          salinasw=Data(:,:,i);
          mx=max(salinasw(:));
          mn=min(salinasw(:));
          I1=uint8(round(((salinasw-mn)/(mx-mn))*255));
          Data2(:,:,i)=I1;
        end   
        end
        %Gaussian
        if k==2
          Data2=[];
        for i=1:204
          salinasw=Data(:,:,i);
          mx=max(salinasw(:));
          mn=min(salinasw(:));
          I1=uint8(round(((salinasw-mn)/(mx-mn))*255));
          J1 = imnoise(I1,'gaussian',0,noise_level(z));
          Data2(:,:,i)=J1;
        end
        mkdir(convertStringsToChars(noise_name(z)));
        cd(convertStringsToChars(noise_name(z)));
        end
        %SaltPepper
        if k==3
          Data2=[];
        for i=1:204
          salinasw=Data(:,:,i);
          mx=max(salinasw(:));
          mn=min(salinasw(:));
          I1=uint8(round(((salinasw-mn)/(mx-mn))*255));
          J1 = imnoise(I1,'salt & pepper',(noise_level(z)*10));
          Data2(:,:,i)=J1;
        end
        mkdir(convertStringsToChars(noise_name(z)));
        cd(convertStringsToChars(noise_name(z)));
        end  
        %Stripe
        if k==4
        Data2=[];
        for i=1:204
        salinasw=Data(:,:,i);
        mx=max(salinasw(:));
        mn=min(salinasw(:));
        I1=uint8(round(((salinasw-mn)/(mx-mn))*255));
        Data2(:,:,i)=I1;
        end
          for n =1:204
            x=(noise_level(z)*1000);
            a=round(x*2.17);
            b=zeros(1,217);
            c= randi([1 217],1,a);
            for m=1:a
             b(1,c(m))=abs(random('Normal',0,1)*255);
            end
            b=uint8(b);
            I(:,:,n)=repmat(b,512,1);
          end
            for n=1:size(I,1)
                 for m=1:size(I,2)
                    for o=1:size(I,3)
                        if I(n,m,o) ~= 0
                            Data2(n,m,o)=I(n,m,o);
                        end
                    end
                 end
            end
        mkdir(convertStringsToChars(noise_name(z)));
        cd(convertStringsToChars(noise_name(z)));
        end
    
    %Pixel to image 
    A1=[];
    A2=[];
    A3=[];
    A4=[];
    A5=[];
    A6=[];
    A7=[];
    A8=[];
    A9=[];
    A10=[];
    A11=[];
    A12=[];
    A13=[];
    A14=[];
    A15=[];
    A16=[];
    cnt1 = 0;
    cnt2 = 0;
    cnt3 = 0;
    cnt4 = 0;
    cnt5 = 0;
    cnt6 = 0;
    cnt7 = 0;
    cnt8 = 0;
    cnt9 = 0;
    cnt10 = 0;
    cnt11 = 0;
    cnt12 = 0;
    cnt13 = 0;
    cnt14 = 0;
    cnt15 = 0;
    cnt16 = 0;
    

    for i=1:512
        for j=1:217
           if label(i,j)==1 
               A1(1+cnt1,:)=Data2(i,j,:);
               cnt1=cnt1+1;
           end
           if label(i,j)==2 
               A2(1+cnt2,:)=Data2(i,j,:);
               cnt2=cnt2+1;
           end
           if label(i,j)==3 
               A3(1+cnt3,:)=Data2(i,j,:);
               cnt3=cnt3+1;
           end
           if label(i,j)==4 
                   A4(1+cnt4,:)=Data2(i,j,:);
                   cnt4=cnt4+1;
           end
           if label(i,j)==5 
                   A5(1+cnt5,:)=Data2(i,j,:);
                   cnt5=cnt5+1;
           end
           if label(i,j)==6 
                   A6(1+cnt6,:)=Data2(i,j,:);
                   cnt6=cnt6+1;
           end
           if label(i,j)==7 
                   A7(1+cnt7,:)=Data2(i,j,:);
                   cnt7=cnt7+1;
           end
           if label(i,j)==8 
                   A8(1+cnt8,:)=Data2(i,j,:);
                   cnt8=cnt8+1;
           end
           if label(i,j)==9 
                   A9(1+cnt9,:)=Data2(i,j,:);
                   cnt9=cnt9+1;
           end
           if label(i,j)==10 
                   A10(1+cnt10,:)=Data2(i,j,:);
                   cnt10=cnt10+1;
           end
           if label(i,j)==11 
                   A11(1+cnt11,:)=Data2(i,j,:);
                   cnt11=cnt11+1;
           end
           if label(i,j)==12 
                   A12(1+cnt12,:)=Data2(i,j,:);
                   cnt12=cnt12+1;
           end
           if label(i,j)==13 
                   A13(1+cnt13,:)=Data2(i,j,:);
                   cnt13=cnt13+1;
           end
           if label(i,j)==14 
                   A14(1+cnt14,:)=Data2(i,j,:);
                   cnt14=cnt14+1;
           end
           if label(i,j)==15 
                   A15(1+cnt15,:)=Data2(i,j,:);
                   cnt15=cnt15+1;
           end
           if label(i,j)==16 
                   A16(1+cnt16,:)=Data2(i,j,:);
                   cnt16=cnt16+1;
           end

        end
    end





    coeff=pca(A1);
    Itransformed = A1*coeff;
    II=Itransformed(:,1:192);
    window1=reshape(II,size(A1,1),192);

    coeff=pca(A2);
    Itransformed = A2*coeff;
    II=Itransformed(:,1:192);
    window2=reshape(II,size(A2,1),192);

    coeff=pca(A3);
    Itransformed = A3*coeff;
    II=Itransformed(:,1:192);
    window3=reshape(II,size(A3,1),192);

    coeff=pca(A4);
    Itransformed = A4*coeff;
    II=Itransformed(:,1:192);
    window4=reshape(II,size(A4,1),192);

    coeff=pca(A5);
    Itransformed = A5*coeff;
    II=Itransformed(:,1:192);
    window5=reshape(II,size(A5,1),192);

    coeff=pca(A6);
    Itransformed = A6*coeff;
    II=Itransformed(:,1:192);
    window6=reshape(II,size(A6,1),192);

    coeff=pca(A7);
    Itransformed = A7*coeff;
    II=Itransformed(:,1:192);
    window7=reshape(II,size(A7,1),192);

    coeff=pca(A8);
    Itransformed = A8*coeff;
    II=Itransformed(:,1:192);
    window8=reshape(II,size(A8,1),192);

    coeff=pca(A9);
    Itransformed = A9*coeff;
    II=Itransformed(:,1:192);
    window9=reshape(II,size(A9,1),192);

    coeff=pca(A10);
    Itransformed = A10*coeff;
    II=Itransformed(:,1:192);
    window10=reshape(II,size(A10,1),192);

    coeff=pca(A11);
    Itransformed = A11*coeff;
    II=Itransformed(:,1:192);
    window11=reshape(II,size(A11,1),192);

    coeff=pca(A12);
    Itransformed = A12*coeff;
    II=Itransformed(:,1:192);
    window12=reshape(II,size(A12,1),192);

    coeff=pca(A13);
    Itransformed = A13*coeff;
    II=Itransformed(:,1:192);
    window13=reshape(II,size(A13,1),192);

    coeff=pca(A14);
    Itransformed = A14*coeff;
    II=Itransformed(:,1:192);
    window14=reshape(II,size(A14,1),192);

    coeff=pca(A15);
    Itransformed = A15*coeff;
    II=Itransformed(:,1:192);
    window15=reshape(II,size(A15,1),192);

    coeff=pca(A16);
    Itransformed = A16*coeff;
    II=Itransformed(:,1:192);
    window16=reshape(II,size(A16,1),192);

    
    mkdir(num2str(1));
    cd(num2str(1))
    for jj=1:size(window1,1)
        D1=window1(jj,1:64);
        dd1=reshape(D1,8,8);
        mx=max(dd1(:));
        mn=min(dd1(:));
        I1=uint8(round(((dd1-mn)/(mx-mn))*255));

        D2=window1(jj,65:128);
        dd2=reshape(D2,8,8);
        mx=max(dd2(:));
        mn=min(dd2(:));
        I2=uint8(round(((dd2-mn)/(mx-mn))*255));

        D3=window1(jj,129:192);
        dd3=reshape(D3,8,8);
        mx=max(dd3(:));
        mn=min(dd3(:));
        I3=uint8(round(((dd3-mn)/(mx-mn))*255));

        FImage= [];
        FImage(:,:,1) = I1;
        FImage(:,:,2) = I2;
        FImage(:,:,3) = I3;
        FImage=uint8(FImage);
        FImage=imresize(FImage,[224,224]);
        filename=[num2str(1),'-',num2str(jj),'.jpg'];
        imwrite(FImage,filename);
    end
    cd ..

    mkdir(num2str(2));
    cd(num2str(2))
    for jj=1:size(window2,1)
        D1=window2(jj,1:64);
        dd1=reshape(D1,8,8);
        mx=max(dd1(:));
        mn=min(dd1(:));
        I1=uint8(round(((dd1-mn)/(mx-mn))*255));

        D2=window2(jj,65:128);
        dd2=reshape(D2,8,8);
        mx=max(dd2(:));
        mn=min(dd2(:));
        I2=uint8(round(((dd2-mn)/(mx-mn))*255));

        D3=window2(jj,129:192);
        dd3=reshape(D3,8,8);
        mx=max(dd3(:));
        mn=min(dd3(:));
        I3=uint8(round(((dd3-mn)/(mx-mn))*255));

        FImage= [];
        FImage(:,:,1) = I1;
        FImage(:,:,2) = I2;
        FImage(:,:,3) = I3;
        FImage=uint8(FImage);
        FImage=imresize(FImage,[224,224]);
        filename=[num2str(2),'-',num2str(jj),'.jpg'];
        imwrite(FImage,filename);
    end
    cd ..

    mkdir(num2str(3));
    cd(num2str(3))
    for jj=1:size(window3,1)
        D1=window3(jj,1:64);
        dd1=reshape(D1,8,8);
        mx=max(dd1(:));
        mn=min(dd1(:));
        I1=uint8(round(((dd1-mn)/(mx-mn))*255));

        D2=window3(jj,65:128);
        dd2=reshape(D2,8,8);
        mx=max(dd2(:));
        mn=min(dd2(:));
        I2=uint8(round(((dd2-mn)/(mx-mn))*255));

        D3=window3(jj,129:192);
        dd3=reshape(D3,8,8);
        mx=max(dd3(:));
        mn=min(dd3(:));
        I3=uint8(round(((dd3-mn)/(mx-mn))*255));

        FImage= [];
        FImage(:,:,1) = I1;
        FImage(:,:,2) = I2;
        FImage(:,:,3) = I3;
        FImage=uint8(FImage);
        FImage=imresize(FImage,[224,224]);
        filename=[num2str(3),'-',num2str(jj),'.jpg'];
        imwrite(FImage,filename);
    end
    cd ..

    mkdir(num2str(4));
    cd(num2str(4))
    for jj=1:size(window4,1)
        D1=window4(jj,1:64);
        dd1=reshape(D1,8,8);
        mx=max(dd1(:));
        mn=min(dd1(:));
        I1=uint8(round(((dd1-mn)/(mx-mn))*255));

        D2=window4(jj,65:128);
        dd2=reshape(D2,8,8);
        mx=max(dd2(:));
        mn=min(dd2(:));
        I2=uint8(round(((dd2-mn)/(mx-mn))*255));

        D3=window4(jj,129:192);
        dd3=reshape(D3,8,8);
        mx=max(dd3(:));
        mn=min(dd3(:));
        I3=uint8(round(((dd3-mn)/(mx-mn))*255));

        FImage= [];
        FImage(:,:,1) = I1;
        FImage(:,:,2) = I2;
        FImage(:,:,3) = I3;
        FImage=uint8(FImage);
        FImage=imresize(FImage,[224,224]);
        filename=[num2str(3),'-',num2str(jj),'.jpg'];
        imwrite(FImage,filename);
    end
    cd ..

    mkdir(num2str(5));
    cd(num2str(5))
    for jj=1:size(window5,1)
        D1=window5(jj,1:64);
        dd1=reshape(D1,8,8);
        mx=max(dd1(:));
        mn=min(dd1(:));
        I1=uint8(round(((dd1-mn)/(mx-mn))*255));

        D2=window5(jj,65:128);
        dd2=reshape(D2,8,8);
        mx=max(dd2(:));
        mn=min(dd2(:));
        I2=uint8(round(((dd2-mn)/(mx-mn))*255));

        D3=window5(jj,129:192);
        dd3=reshape(D3,8,8);
        mx=max(dd3(:));
        mn=min(dd3(:));
        I3=uint8(round(((dd3-mn)/(mx-mn))*255));

        FImage= [];
        FImage(:,:,1) = I1;
        FImage(:,:,2) = I2;
        FImage(:,:,3) = I3;
        FImage=uint8(FImage);
        FImage=imresize(FImage,[224,224]);
        filename=[num2str(3),'-',num2str(jj),'.jpg'];
        imwrite(FImage,filename);
    end
    cd ..

    mkdir(num2str(6));
    cd(num2str(6))
    for jj=1:size(window6,1)
        D1=window6(jj,1:64);
        dd1=reshape(D1,8,8);
        mx=max(dd1(:));
        mn=min(dd1(:));
        I1=uint8(round(((dd1-mn)/(mx-mn))*255));

        D2=window6(jj,65:128);
        dd2=reshape(D2,8,8);
        mx=max(dd2(:));
        mn=min(dd2(:));
        I2=uint8(round(((dd2-mn)/(mx-mn))*255));

        D3=window6(jj,129:192);
        dd3=reshape(D3,8,8);
        mx=max(dd3(:));
        mn=min(dd3(:));
        I3=uint8(round(((dd3-mn)/(mx-mn))*255));

        FImage= [];
        FImage(:,:,1) = I1;
        FImage(:,:,2) = I2;
        FImage(:,:,3) = I3;
        FImage=uint8(FImage);
        FImage=imresize(FImage,[224,224]);
        filename=[num2str(3),'-',num2str(jj),'.jpg'];
        imwrite(FImage,filename);
    end
    cd ..

    mkdir(num2str(7));
    cd(num2str(7))
    for jj=1:size(window7,1)
        D1=window7(jj,1:64);
        dd1=reshape(D1,8,8);
        mx=max(dd1(:));
        mn=min(dd1(:));
        I1=uint8(round(((dd1-mn)/(mx-mn))*255));

        D2=window7(jj,65:128);
        dd2=reshape(D2,8,8);
        mx=max(dd2(:));
        mn=min(dd2(:));
        I2=uint8(round(((dd2-mn)/(mx-mn))*255));

        D3=window7(jj,129:192);
        dd3=reshape(D3,8,8);
        mx=max(dd3(:));
        mn=min(dd3(:));
        I3=uint8(round(((dd3-mn)/(mx-mn))*255));

        FImage= [];
        FImage(:,:,1) = I1;
        FImage(:,:,2) = I2;
        FImage(:,:,3) = I3;
        FImage=uint8(FImage);
        FImage=imresize(FImage,[224,224]);
        filename=[num2str(3),'-',num2str(jj),'.jpg'];
        imwrite(FImage,filename);
    end
    cd ..

    mkdir(num2str(8));
    cd(num2str(8))
    for jj=1:size(window8,1)
        D1=window8(jj,1:64);
        dd1=reshape(D1,8,8);
        mx=max(dd1(:));
        mn=min(dd1(:));
        I1=uint8(round(((dd1-mn)/(mx-mn))*255));

        D2=window8(jj,65:128);
        dd2=reshape(D2,8,8);
        mx=max(dd2(:));
        mn=min(dd2(:));
        I2=uint8(round(((dd2-mn)/(mx-mn))*255));

        D3=window8(jj,129:192);
        dd3=reshape(D3,8,8);
        mx=max(dd3(:));
        mn=min(dd3(:));
        I3=uint8(round(((dd3-mn)/(mx-mn))*255));

        FImage= [];
        FImage(:,:,1) = I1;
        FImage(:,:,2) = I2;
        FImage(:,:,3) = I3;
        FImage=uint8(FImage);
        FImage=imresize(FImage,[224,224]);
        filename=[num2str(3),'-',num2str(jj),'.jpg'];
        imwrite(FImage,filename);
    end
    cd ..

    mkdir(num2str(9));
    cd(num2str(9))
    for jj=1:size(window9,1)
        D1=window9(jj,1:64);
        dd1=reshape(D1,8,8);
        mx=max(dd1(:));
        mn=min(dd1(:));
        I1=uint8(round(((dd1-mn)/(mx-mn))*255));

        D2=window9(jj,65:128);
        dd2=reshape(D2,8,8);
        mx=max(dd2(:));
        mn=min(dd2(:));
        I2=uint8(round(((dd2-mn)/(mx-mn))*255));

        D3=window9(jj,129:192);
        dd3=reshape(D3,8,8);
        mx=max(dd3(:));
        mn=min(dd3(:));
        I3=uint8(round(((dd3-mn)/(mx-mn))*255));

        FImage= [];
        FImage(:,:,1) = I1;
        FImage(:,:,2) = I2;
        FImage(:,:,3) = I3;
        FImage=uint8(FImage);
        FImage=imresize(FImage,[224,224]);
        filename=[num2str(3),'-',num2str(jj),'.jpg'];
        imwrite(FImage,filename);
    end
    cd ..
    
    mkdir(num2str(10));
    cd(num2str(10))
    for jj=1:size(window10,1)
        D1=window10(jj,1:64);
        dd1=reshape(D1,8,8);
        mx=max(dd1(:));
        mn=min(dd1(:));
        I1=uint8(round(((dd1-mn)/(mx-mn))*255));

        D2=window10(jj,65:128);
        dd2=reshape(D2,8,8);
        mx=max(dd2(:));
        mn=min(dd2(:));
        I2=uint8(round(((dd2-mn)/(mx-mn))*255));

        D3=window10(jj,129:192);
        dd3=reshape(D3,8,8);
        mx=max(dd3(:));
        mn=min(dd3(:));
        I3=uint8(round(((dd3-mn)/(mx-mn))*255));

        FImage= [];
        FImage(:,:,1) = I1;
        FImage(:,:,2) = I2;
        FImage(:,:,3) = I3;
        FImage=uint8(FImage);
        FImage=imresize(FImage,[224,224]);
        filename=[num2str(3),'-',num2str(jj),'.jpg'];
        imwrite(FImage,filename);
    end
    cd ..
    
    mkdir(num2str(11));
    cd(num2str(11))
    for jj=1:size(window11,1)
        D1=window11(jj,1:64);
        dd1=reshape(D1,8,8);
        mx=max(dd1(:));
        mn=min(dd1(:));
        I1=uint8(round(((dd1-mn)/(mx-mn))*255));

        D2=window11(jj,65:128);
        dd2=reshape(D2,8,8);
        mx=max(dd2(:));
        mn=min(dd2(:));
        I2=uint8(round(((dd2-mn)/(mx-mn))*255));

        D3=window11(jj,129:192);
        dd3=reshape(D3,8,8);
        mx=max(dd3(:));
        mn=min(dd3(:));
        I3=uint8(round(((dd3-mn)/(mx-mn))*255));

        FImage= [];
        FImage(:,:,1) = I1;
        FImage(:,:,2) = I2;
        FImage(:,:,3) = I3;
        FImage=uint8(FImage);
        FImage=imresize(FImage,[224,224]);
        filename=[num2str(3),'-',num2str(jj),'.jpg'];
        imwrite(FImage,filename);
    end
    cd ..
    
    mkdir(num2str(12));
    cd(num2str(12))
    for jj=1:size(window12,1)
        D1=window12(jj,1:64);
        dd1=reshape(D1,8,8);
        mx=max(dd1(:));
        mn=min(dd1(:));
        I1=uint8(round(((dd1-mn)/(mx-mn))*255));

        D2=window12(jj,65:128);
        dd2=reshape(D2,8,8);
        mx=max(dd2(:));
        mn=min(dd2(:));
        I2=uint8(round(((dd2-mn)/(mx-mn))*255));

        D3=window12(jj,129:192);
        dd3=reshape(D3,8,8);
        mx=max(dd3(:));
        mn=min(dd3(:));
        I3=uint8(round(((dd3-mn)/(mx-mn))*255));

        FImage= [];
        FImage(:,:,1) = I1;
        FImage(:,:,2) = I2;
        FImage(:,:,3) = I3;
        FImage=uint8(FImage);
        FImage=imresize(FImage,[224,224]);
        filename=[num2str(3),'-',num2str(jj),'.jpg'];
        imwrite(FImage,filename);
    end
    cd ..
    
    mkdir(num2str(13));
    cd(num2str(13))
    for jj=1:size(window13,1)
        D1=window13(jj,1:64);
        dd1=reshape(D1,8,8);
        mx=max(dd1(:));
        mn=min(dd1(:));
        I1=uint8(round(((dd1-mn)/(mx-mn))*255));

        D2=window13(jj,65:128);
        dd2=reshape(D2,8,8);
        mx=max(dd2(:));
        mn=min(dd2(:));
        I2=uint8(round(((dd2-mn)/(mx-mn))*255));

        D3=window13(jj,129:192);
        dd3=reshape(D3,8,8);
        mx=max(dd3(:));
        mn=min(dd3(:));
        I3=uint8(round(((dd3-mn)/(mx-mn))*255));

        FImage= [];
        FImage(:,:,1) = I1;
        FImage(:,:,2) = I2;
        FImage(:,:,3) = I3;
        FImage=uint8(FImage);
        FImage=imresize(FImage,[224,224]);
        filename=[num2str(3),'-',num2str(jj),'.jpg'];
        imwrite(FImage,filename);
    end
    cd ..
    
    mkdir(num2str(14));
    cd(num2str(14))
    for jj=1:size(window14,1)
        D1=window14(jj,1:64);
        dd1=reshape(D1,8,8);
        mx=max(dd1(:));
        mn=min(dd1(:));
        I1=uint8(round(((dd1-mn)/(mx-mn))*255));

        D2=window14(jj,65:128);
        dd2=reshape(D2,8,8);
        mx=max(dd2(:));
        mn=min(dd2(:));
        I2=uint8(round(((dd2-mn)/(mx-mn))*255));

        D3=window14(jj,129:192);
        dd3=reshape(D3,8,8);
        mx=max(dd3(:));
        mn=min(dd3(:));
        I3=uint8(round(((dd3-mn)/(mx-mn))*255));

        FImage= [];
        FImage(:,:,1) = I1;
        FImage(:,:,2) = I2;
        FImage(:,:,3) = I3;
        FImage=uint8(FImage);
        FImage=imresize(FImage,[224,224]);
        filename=[num2str(3),'-',num2str(jj),'.jpg'];
        imwrite(FImage,filename);
    end
    cd ..
    
    mkdir(num2str(15));
    cd(num2str(15))
    for jj=1:size(window15,1)
        D1=window15(jj,1:64);
        dd1=reshape(D1,8,8);
        mx=max(dd1(:));
        mn=min(dd1(:));
        I1=uint8(round(((dd1-mn)/(mx-mn))*255));

        D2=window15(jj,65:128);
        dd2=reshape(D2,8,8);
        mx=max(dd2(:));
        mn=min(dd2(:));
        I2=uint8(round(((dd2-mn)/(mx-mn))*255));

        D3=window15(jj,129:192);
        dd3=reshape(D3,8,8);
        mx=max(dd3(:));
        mn=min(dd3(:));
        I3=uint8(round(((dd3-mn)/(mx-mn))*255));

        FImage= [];
        FImage(:,:,1) = I1;
        FImage(:,:,2) = I2;
        FImage(:,:,3) = I3;
        FImage=uint8(FImage);
        FImage=imresize(FImage,[224,224]);
        filename=[num2str(3),'-',num2str(jj),'.jpg'];
        imwrite(FImage,filename);
    end
    cd ..
    
    mkdir(num2str(16));
    cd(num2str(16))
    for jj=1:size(window16,1)
        D1=window16(jj,1:64);
        dd1=reshape(D1,8,8);
        mx=max(dd1(:));
        mn=min(dd1(:));
        I1=uint8(round(((dd1-mn)/(mx-mn))*255));

        D2=window16(jj,65:128);
        dd2=reshape(D2,8,8);
        mx=max(dd2(:));
        mn=min(dd2(:));
        I2=uint8(round(((dd2-mn)/(mx-mn))*255));

        D3=window16(jj,129:192);
        dd3=reshape(D3,8,8);
        mx=max(dd3(:));
        mn=min(dd3(:));
        I3=uint8(round(((dd3-mn)/(mx-mn))*255));

        FImage= [];
        FImage(:,:,1) = I1;
        FImage(:,:,2) = I2;
        FImage(:,:,3) = I3;
        FImage=uint8(FImage);
        FImage=imresize(FImage,[224,224]);
        filename=[num2str(3),'-',num2str(jj),'.jpg'];
        imwrite(FImage,filename);
    end
    cd ..
    if k==1
        break;
    end
    if k~=1
        cd ..
    end
end
cd ..
end
