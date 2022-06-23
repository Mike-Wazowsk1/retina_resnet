import PySimpleGUI as sg
from visual.stats import plotti

hist = ['a']
layout = [
    [sg.Text('Name'), sg.InputText(),
     ],
    [sg.Text('Group'), sg.InputText(),
     ],

    [sg.Text('History'), sg.Output(size=(42,5))],
    [sg.Submit(), sg.Cancel()]

]

window = sg.Window('Stats', layout)

while True:  # The Event Loop

    event, values = window.read()
    if event in (None, 'Exit', 'Cancel'):
        break
    if event == 'Submit':
        if len(values[0]) == 0:
            name = None
        else:
            name = values[0]
        if len(values[1]) == 0:
            group = None
        else:
            group = values[1]
        print(name if name is not None else "", group if group is not None else '')
        plotti(name, group)
