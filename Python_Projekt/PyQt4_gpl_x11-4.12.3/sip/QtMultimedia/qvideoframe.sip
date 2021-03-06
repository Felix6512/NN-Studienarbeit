// qvideoframe.sip generated by MetaSIP
//
// This file is part of the QtMultimedia Python extension module.
//
// Copyright (c) 2018 Riverbank Computing Limited <info@riverbankcomputing.com>
// 
// This file is part of PyQt4.
// 
// This file may be used under the terms of the GNU General Public License
// version 3.0 as published by the Free Software Foundation and appearing in
// the file LICENSE included in the packaging of this file.  Please review the
// following information to ensure the GNU General Public License version 3.0
// requirements will be met: http://www.gnu.org/copyleft/gpl.html.
// 
// If you do not wish to use this file under the terms of the GPL version 3.0
// then you may purchase a commercial license.  For more information contact
// info@riverbankcomputing.com.
// 
// This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
// WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.


%If (Qt_4_6_0 -)

class QVideoFrame
{
%TypeHeaderCode
#include <qvideoframe.h>
%End

public:
    enum FieldType
    {
        ProgressiveFrame,
        TopField,
        BottomField,
        InterlacedFrame,
    };

    enum PixelFormat
    {
        Format_Invalid,
        Format_ARGB32,
        Format_ARGB32_Premultiplied,
        Format_RGB32,
        Format_RGB24,
        Format_RGB565,
        Format_RGB555,
        Format_ARGB8565_Premultiplied,
        Format_BGRA32,
        Format_BGRA32_Premultiplied,
        Format_BGR32,
        Format_BGR24,
        Format_BGR565,
        Format_BGR555,
        Format_BGRA5658_Premultiplied,
        Format_AYUV444,
        Format_AYUV444_Premultiplied,
        Format_YUV444,
        Format_YUV420P,
        Format_YV12,
        Format_UYVY,
        Format_YUYV,
        Format_NV12,
        Format_NV21,
        Format_IMC1,
        Format_IMC2,
        Format_IMC3,
        Format_IMC4,
        Format_Y8,
        Format_Y16,
%If (Qt_5_0_0 -)
        Format_Jpeg,
%End
%If (Qt_5_0_0 -)
        Format_CameraRaw,
%End
%If (Qt_5_0_0 -)
        Format_AdobeDng,
%End
        Format_User,
    };

    QVideoFrame();
    QVideoFrame(QAbstractVideoBuffer *buffer, const QSize &size, QVideoFrame::PixelFormat format);
    QVideoFrame(int bytes, const QSize &size, int bytesPerLine, QVideoFrame::PixelFormat format);
    QVideoFrame(const QImage &image);
    QVideoFrame(const QVideoFrame &other);
    ~QVideoFrame();
    bool isValid() const;
    QVideoFrame::PixelFormat pixelFormat() const;
    QAbstractVideoBuffer::HandleType handleType() const;
    QSize size() const;
    int width() const;
    int height() const;
    QVideoFrame::FieldType fieldType() const;
    void setFieldType(QVideoFrame::FieldType);
    bool isMapped() const;
    bool isReadable() const;
    bool isWritable() const;
    QAbstractVideoBuffer::MapMode mapMode() const;
    bool map(QAbstractVideoBuffer::MapMode mode);
    void unmap();
    int bytesPerLine() const;
    SIP_PYOBJECT bits() /TypeHint="sip.voidptr"/;
%MethodCode
        uchar *mem;
        
        Py_BEGIN_ALLOW_THREADS
        mem = sipCpp->bits();
        Py_END_ALLOW_THREADS
        
        if (mem)
        {
            sipRes = sipConvertFromVoidPtrAndSize(mem, sipCpp->mappedBytes());
        }
        else
        {
            sipRes = Py_None;
            Py_INCREF(sipRes);
        }
%End

    int mappedBytes() const;
    QVariant handle() const;
    qint64 startTime() const;
    void setStartTime(qint64 time);
    qint64 endTime() const;
    void setEndTime(qint64 time);
    static QVideoFrame::PixelFormat pixelFormatFromImageFormat(QImage::Format format);
    static QImage::Format imageFormatFromPixelFormat(QVideoFrame::PixelFormat format);
%If (Qt_5_0_0 -)
    QVariantMap availableMetaData() const;
%End
%If (Qt_5_0_0 -)
    QVariant metaData(const QString &key) const;
%End
%If (Qt_5_0_0 -)
    void setMetaData(const QString &key, const QVariant &value);
%End
};

%End
%If (Qt_4_6_0 -)
// Mapped type for QList<QVideoFrame::PixelFormat>.
%MappedType QList<QVideoFrame::PixelFormat> /TypeHint="List[QVideoFrame.PixelFormat]", TypeHintValue="[]"/
{
%TypeHeaderCode
#include <qvideoframe.h>
%End

%ConvertFromTypeCode
    // Create the list.
    PyObject *l;

    if ((l = PyList_New(sipCpp->size())) == NULL)
        return NULL;

    // Set the list elements.
    for (int i = 0; i < sipCpp->size(); ++i)
    {
        QVideoFrame::PixelFormat e = sipCpp->at(i);
        PyObject *eobj;

        if ((eobj = sipConvertFromEnum(e, sipType_QVideoFrame_PixelFormat)) == NULL)
        {
            Py_DECREF(l);

            return NULL;
        }

        PyList_SET_ITEM(l, i, eobj);
    }

    return l;
%End

%ConvertToTypeCode
    // Check the type if that is all that is required.
    if (sipIsErr == NULL)
    {
        if (!PyList_Check(sipPy))
            return 0;

        for (SIP_SSIZE_T i = 0; i < PyList_GET_SIZE(sipPy); ++i)
            if (!sipCanConvertToEnum(PyList_GET_ITEM(sipPy, i), sipType_QVideoFrame_PixelFormat))
                return 0;

        return 1;
    }

    QList<QVideoFrame::PixelFormat> *ql = new QList<QVideoFrame::PixelFormat>;
 
    for (SIP_SSIZE_T i = 0; i < PyList_GET_SIZE(sipPy); ++i)
    {
        long l = SIPLong_AsLong(PyList_GET_ITEM(sipPy, i));
        ql->append(static_cast<QVideoFrame::PixelFormat>(l));
    }
 
    *sipCppPtr = ql;
 
    return sipGetState(sipTransferObj);
%End
};
%End
