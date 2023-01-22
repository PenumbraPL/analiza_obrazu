import wx
from ocr_lib import ocr 


class MyFrame(wx.Frame):

    def __init__(self, *args, **kw):
        super(MyFrame, self).__init__(*args, **kw)

        pnl = wx.Panel(self)

        self.text = wx.TextCtrl(pnl, size=wx.Size(400, 400), style=wx.TE_MULTILINE|wx.TE_READONLY)
        self.text.SetDefaultStyle(wx.TextAttr(wx.RED))
        self.text.AppendText("Text detected:\n")
        self.text.SetDefaultStyle(wx.TextAttr(wx.NullColour, wx.LIGHT_GREY))
        self.text.AppendText("----------------------\n")
        self.text.SetDefaultStyle(wx.TextAttr(wx.BLUE))

        st = wx.StaticText(pnl, label="Welcome!")
        font = st.GetFont()
        font.PointSize += 10
        font = font.Bold()
        st.SetFont(font)


        img = wx.Image(800,800)
        self.imageCtrl = wx.StaticBitmap(pnl, wx.ID_ANY, wx.Bitmap(img))


        image_sizer = wx.BoxSizer(wx.VERTICAL)
        image_sizer.Add(st, wx.SizerFlags().Border(wx.TOP|wx.LEFT, 25))
        image_sizer.Add(self.imageCtrl,  wx.SizerFlags().Border(wx.ALL))

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.text, wx.SizerFlags().Border())
        sizer.Add(image_sizer, wx.SizerFlags().Border(wx.ALL))
        pnl.SetSizer(sizer)

        self.makeMenuBar()

        self.CreateStatusBar()
        self.SetStatusText("Text recognition!")

        

    def makeMenuBar(self):
        fileMenu = wx.Menu()
        loadImage = fileMenu.Append(-1, "&Load Image...\tCtrl-H",
                "Load image desired to be processed")
        fileMenu.AppendSeparator()
        exitItem = fileMenu.Append(wx.ID_EXIT)
        helpMenu = wx.Menu()
        aboutItem = helpMenu.Append(wx.ID_ABOUT)

        menuBar = wx.MenuBar()
        menuBar.Append(fileMenu, "&File")
        menuBar.Append(helpMenu, "&Help")

        self.SetMenuBar(menuBar)

        self.Bind(wx.EVT_MENU, self.OnOpen, loadImage)
        self.Bind(wx.EVT_MENU, self.OnExit,  exitItem)
        self.Bind(wx.EVT_MENU, self.OnAbout, aboutItem)


    def OnExit(self, event):
        self.Close(True)


    def LoadImage(self, filepath):
        print('Loading Image:', filepath)
        
        letter_list, img = ocr(filepath)
        print(letter_list)
        letter_list.append('\n')
        wx_img = wx.ImageFromBuffer(img.shape[1], img.shape[0], img)
        self.imageCtrl.SetBitmap(wx.Bitmap(wx_img))
        self.text.AppendText(''.join(letter_list))

    def OnOpen(self, event):
        with wx.FileDialog(self, "Open PNG file", wildcard="PNG files (*.png)|*.png",
                        style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:

            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return     # the user changed their mind

            pathname = fileDialog.GetPath()
            self.LoadImage(pathname)
        fileDialog.Destroy()

    def OnAbout(self, event):
        wx.MessageBox("Created by Jan Bizoń",
                      "wxPython, Python=3.7.15",
                      wx.OK|wx.ICON_INFORMATION)


if __name__ == '__main__':
    app = wx.App()
    frm = MyFrame(None, title='Text Recognition', pos=wx.Point(100, 100), size=wx.Size(1800, 800))
    frm.LoadImage("./data/ala_ma_kota2.png")
    frm.Show()
    app.MainLoop()