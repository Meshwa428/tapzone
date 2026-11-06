"""tapzones: tap the screen, the mic hears where (docs/usage.md).

    tapzones run                    the background service (tapzones.service)
    tapzones status | pause | resume | toggle
    tapzones mistake                the last tap wasn't one: keep it as noise
    tapzones confirm SECTION        the last sound was a tap there: run it, keep it
    tapzones layout                 edit each gesture's sections, save and train
    tapzones flash                  the GTK flash feedback (desktops without Omarchy)
    tapzones setup                  calibrate this device: tap the dots, then everyday noise
    tapzones calibrate | test | free | locate | collect | noise
                                    the overlay: record taps, see and test the models
    tapzones train sections GESTURE | gate | final | xy | signed
"""
import argparse, sys


def main(argv=None):
    p = argparse.ArgumentParser(prog="tapzones", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n\n", 1)[1])
    sub = p.add_subparsers(dest="cmd", required=True, metavar="COMMAND")
    sub.add_parser("run", help="the background service: hears taps, runs their commands")
    for cmd, what in (("status", "what the service is doing, as JSON"), ("pause", "stop running commands"),
                      ("resume", "run commands again"), ("toggle", "pause or resume"),
                      ("mistake", "the last tap wasn't one: teach the noise filter (bind it to a key)")):
        sub.add_parser(cmd, help=what)
    s = sub.add_parser("confirm", help="the last sound (a miss, or a tap in the wrong section) was a tap in SECTION: run it, keep it")
    s.add_argument("section", help="the section's name (\"top left\") or number, from 1")
    sub.add_parser("layout", help="the layout editor (GTK; on Omarchy, click the bar icon)")
    sub.add_parser("flash", help="the GTK flash feedback, for desktops without the Omarchy plugin")

    sub.add_parser("setup", help="overlay: this device's calibration, ~3 min (the panel starts it)")
    sub.add_parser("calibrate", help="overlay: 15 taps per quadrant, then review")
    for mode, what in (("test", "overlay: tap 20 random points, see what each was heard as"),
                       ("free", "overlay: tap anywhere; N marks the last sound as not a tap")):
        s = sub.add_parser(mode, help=what)
        s.add_argument("--train", metavar="SESSION,…", help="an SVM trained on these sessions in the data folder")
        s.add_argument("--signed", action="store_true", help="the signed CNN (tapzones train signed)")
        s.add_argument("--knn", action="store_true", help="the calibration kNN, not the final ensemble")
    sub.add_parser("locate", help="overlay: tap where shown; the x,y model draws where it heard you")
    s = sub.add_parser("collect", help="overlay: record balanced training taps (docs/training.md)")
    s.add_argument("per_spot", nargs="?", type=int, help="taps per grid spot (default 5)")
    s.add_argument("per_zone", nargs="?", type=int, help="random points per quadrant (default 30)")
    sub.add_parser("noise", help="overlay: record sounds that are NOT screen taps, for the gate")

    s = sub.add_parser("train", help="train a model from the recordings")
    jobs = s.add_subparsers(dest="job", required=True, metavar="MODEL")
    j = jobs.add_parser("sections", help="a gesture's section model, for its layout in gestures.toml")
    j.add_argument("gesture", choices=["single", "double", "triple", "two_finger", "three_finger"])
    for job, what in (("gate", "tap or not"), ("final", "the overlay's quadrant ensemble"),
                      ("xy", "rough x,y position"), ("signed", "the signed CNN")):
        jobs.add_parser(job, help=what)
    a = p.parse_args(argv)

    # Imported per command: the GTK ones restart the process with layer
    # shell preloaded, which the headless ones must not.
    if a.cmd == "run":
        from .service import engine
        return engine.run()
    if a.cmd in ("status", "pause", "resume", "toggle", "mistake"):
        from .service import client
        return client.send(a.cmd)
    if a.cmd == "confirm":
        from .service import client
        return client.send("confirm", a.section)
    if a.cmd == "train":
        from . import train
        return train.export_sections(a.gesture) if a.job == "sections" else train.JOBS[a.job]()
    if a.cmd == "layout":
        from .gtk import editor
        return editor.main()
    if a.cmd == "flash":
        from .gtk import flash
        return flash.main()
    from .gtk import overlay
    if a.cmd == "collect":
        return overlay.main("collect" + "".join(f":{n}" for n in (a.per_spot, a.per_zone) if n is not None))
    if a.cmd in ("test", "free"):
        train = ["signed"] if a.signed else a.train.split(",") if a.train else None
        return overlay.main(a.cmd, train, a.knn)
    return overlay.main(a.cmd)


if __name__ == "__main__":
    sys.exit(main())
