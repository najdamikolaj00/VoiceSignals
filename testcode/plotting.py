'''

Program for plotting waveplots and spectograms

'''
import librosa
import librosa.display as dis
import numpy as np
import matplotlib.pyplot as plt
import os 
import time

def spectogram(audio_path, freq_choice):
    x, sr = librosa.load(audio_path)
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(np.abs(X), ref = np.max)
    
    fig, ax = plt.subplots()
    img = dis.specshow(Xdb, x_axis = 'time', y_axis = freq_choice, ax = ax)
    ax.set(title = 'Spectogram for ' + str(sr) + ' sampling rate (Hz)')
    fig.colorbar(img, ax = ax, format = "%+2.f dB")
    plt.show()
    

def waveplot(plot_type, list_of_paths, list_of_names, list_of_moods):
    dict_of_colors = {'b' : 'blue', 'g' : 'green', 'r' : 'red', 'c' : 'cyan', 'm' : 'magenta', 'y' : 'yellow', 'k' : 'black', 'w' : 'white'}

    if plot_type == 0:

        plt.figure()
        for i in range(len(list_of_paths)):
            print('\ncolor for ', i + 1, ' plot, avaiable: ', dict_of_colors)
            color_type = input('\ncolor: ')
            x, sr = librosa.load(list_of_paths[i])
            dis.waveplot(x, sr = sr, color = color_type, label = list_of_names[i] + ' ' + list_of_moods[i]) 
        plt.grid(True)
        plt.xlabel('time [s]')
        plt.ylabel('displacement')
        plt.title('Audio')
        plt.legend()
        plt.show()
    elif plot_type == 1:
        plt.figure()
        if len(list_of_paths) % 3 == 0:
            nrows = 3
            ncols = len(list_of_paths)//3
        elif len(list_of_paths) < 3:
            nrows = len(list_of_paths)
            ncols = 1
        elif (len(list_of_paths) % 3 != 0  and len(list_of_paths) > 3):
            nrows = 3
            ncols = len(list_of_paths)//3 + 1
        for i in range(len(list_of_paths)):
            print('\ncolor for ', i + 1, ' plot, avaiable: ', dict_of_colors)
            color_type = input('\ncolor: ')
            x, sr = librosa.load(list_of_paths[i])
            plt.subplot(nrows, ncols, i + 1)
            dis.waveplot(x, sr = sr, color = color_type, label = list_of_names[i] +list_of_moods[i])
            plt.grid(True)
            plt.xlabel('time [s]')
            plt.ylabel('displacement')
            plt.legend()
        plt.show()

def main():

    choice = int(input('\nWould you like to plot waveplot (0) or spectogram(1)?: '))

    if choice == 0:
        num_of_plots = int(input('\nNumber of plots: '))
        if num_of_plots != 1:
            plot_type = int(input('\nSame figure (0), \ndifferent figures (1) \nchoice: '))
            if not (plot_type == 0 or plot_type == 1):
                time.sleep(2)
                raise ValueError('Error, wrong number!!!')
        else:
            plot_type = 0
    
        list_of_paths = []
        list_of_names = []
        list_of_moods = []
        

        for i in range(num_of_plots):
            name = input('\nname: ')
            list_of_names.append(name)
            mood = input('\nmood: ')
            list_of_moods.append(mood)
            sample_num = input('\nnumber of sample (from 1 to 4): ')
            path = 'Datano2/' + str(mood) + '/'
            data = str(name) + str(mood) + str(sample_num) + '.wav'
            path = os.path.join(path, data)
            list_of_paths.append(path)
        
        waveplot(plot_type, list_of_paths, list_of_names, list_of_moods)
        pass
    elif choice == 1:
        name = input('\nname: ')
        mood = input('\nmood: ')
        sample_num = input('\nnumber of sample (from 1 to 4): ')
        freq_choice = input('\nwhat type of frequency scale would you like to get linear or logarithmic (log)?: ')
        if (freq_choice is not 'linear' or freq_choice is not 'log'):
            raise ValueError('Error, wrong number!!!')
        else:
            audio_path = 'Datano2/' + str(mood) + '/' + str(name) + str(mood) + str(sample_num) + '.wav'
            spectogram(audio_path, freq_choice)

if __name__ == '__main__':
    main()





