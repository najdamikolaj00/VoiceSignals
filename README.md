<h2>Freelancer's project - analysis of human voice</h2>
<h4>The main goal of this project is to create the tool for sobriety detection.</h4>

<h3>Tools, languages and the most used library:</h3>
<ul>
    <li>Visual Studio Code</li>
    <li>Python 3</li>
    <li>Librosa https://librosa.org/doc/latest/index.html</li>
</ul>

<h3>To do list:</h3>
<ul>
    <li>Collecting more data and learn how to use SQL to do it.</li>
    <li>Make a clear way to use of the program.</li>
    <li>Make the code more readable.</li>
    <li>Learn about Machine Learning in case of using it for this project.</li>
</ul>

<h3> <strong>Diagram</strong> shows how our program works.</h3>
<p style="text-align:center;"><img src = "testcode/photos/voicesignalsdiagram.png" width = "400" ></p>










***

To see what is represented by the image just hover the cursor over the image.

***
**Audio vizualization**

***
<img  title = "Sober" src = "Code/photos/wavesober.png" width = "600" >

<img title = "Unsober" src = "Code/photos/waveunsober.png" width = "600" >

***

**Spectograms** --> a visual way of representing the signal strength. - https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html

***
<img title = "Left - sober, Right - unsober" src = "Code/photos/spectograms.png" width = "600" >


***

**Spectral centroid** --> It indicates where the center of mass of the spectrum is located. - https://en.wikipedia.org/wiki/Spectral_centroid

***
<img title = "Sober" src = "Code/photos/spectralcentroidsober.png" width = "600" >

<img title = "Unsober" src = "Code/photos/spectralcentroidunsober.png" width = "600" >


***

**Spectral bandwidth** --> shows how our program works. - The spectral bandwidth is defined as the width of the band of light at one-half the peak maximum (or full width at half maximum [FWHM]) and is represented by the two vertical red lines and λSB on the wavelength axis. - https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html

***
<img title = "Sober" src = "Code/photos/spectralbandwidthsober.png" width = "600" >

<img title = "Unsober" src = "Code/photos/spectralbandwidthunsober.png" width = "600" >


***

**Spectral rolloff** --> shows how our program works. - Spectral rolloff indicates the roll off frequency for each frame in signal. - https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html

***
<img title = "Sober" src = "Code/photos/spectralrolloffsober.png" width = "600" >

<img title = "Unsober" src = "Code/photos/spectralrolloffunsober.png" width = "600" >


***

**MFCCGs** --> shows how our program works. - Small set of features which concisely describe the overall shape of a spectral envelope.

***
<img title = "Sober" src = "Code/photos/mfsober.png" width = "600" >

<img title = "Unsober" src = "Code/photos/mfunsober.png" width = "600" >


***

**Chroma feature** --> shows how our program works. - It provides a strong way to describe a similarity measure between voice pieces.

***
<img title = "Sober" src = "Code/photos/cfsober.png" width = "600" >

<img title = "Unsober" src = "Code/photos/cfunsober.png" width = "600" >

