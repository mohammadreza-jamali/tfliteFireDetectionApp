import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite_v2/tflite_v2.dart';

late List<CameraDescription> _cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  _cameras = await availableCameras();
  runApp(const CameraScreen());
}

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  late CameraController controller;
  CameraImage? img;
  bool isBusy = false;
  String result = "";

  List<dynamic>? _recognitions;
  @override
  void initState() {
    super.initState();
    loadModel();

    controller = CameraController(_cameras[0], ResolutionPreset.high);
    controller.initialize().then((_) {
      if (!mounted) {
        return;
      }
      controller.startImageStream((image) {
        if (!isBusy) {
          isBusy = true;
          img = image;
          classifyImage();
        }
      });
      setState(() {});
      isBusy = false;
    }).catchError((Object e) {
      if (e is CameraException) {
        switch (e.code) {
          case 'CameraAccessDenied':
            print('User denied camera access.');
            break;
          default:
            print('Handle other errors.');
            break;
        }
      }
    });
  }

  Future<void> loadModel() async {
    await Tflite.loadModel(
        model: "assets/model_unquant.tflite",
        labels: "assets/labels_unquant.txt",
        numThreads: 1, // defaults to 1
        isAsset: true, // defaults to true, set to false to load resources outside assets
        useGpuDelegate: false // defaults to false, set to true to use GPU delegate
        );
    debugPrint("model loaded");
  }

  Future<void> classifyImage() async {
    if (img == null) return;
    var recognitions = await Tflite.runModelOnFrame(
        bytesList: img!.planes.map((plane) {
          return plane.bytes;
        }).toList(),
        imageHeight: img!.height,
        imageWidth: img!.width,
        imageMean: 127.5, // defaults to 127.5
        imageStd: 127.5, // defaults to 127.5
        rotation: 90, // defaults to 90, Android only
        numResults: 3, // defaults to 5
        threshold: 0.8, // defaults to 0.1
        asynch: true // defaults to true
        );

    setState(() {
      _recognitions = recognitions;
      var t = _recognitions?.map((res) {
        return "${res["label"].toString().substring(2)}: ${(res["confidence"] * 100).toStringAsFixed(2)}%";
      });
      result = t?.join("\n") ?? "";
    });
    isBusy = false;
  }

  @override
  void dispose() {
    controller.dispose();
    Tflite.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!controller.value.isInitialized) {
      return Container();
    }
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: Scaffold(
        body: Stack(
          fit: StackFit.expand,
          children: [
            CameraPreview(controller),
            Container(
              margin: const EdgeInsets.only(left: 10, bottom: 10),
              child: Align(
                alignment: Alignment.bottomLeft,
                child: Text(
                  result,
                  style: const TextStyle(color: Colors.white, fontSize: 25),
                ),
              ),
            )
          ],
        ),
      ),
    );
  }
}
